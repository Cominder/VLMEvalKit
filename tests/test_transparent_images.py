"""Regression tests for transparent images in the smp image codecs (issue 1752)."""
import base64
import importlib.util
import io
import sys
import types
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


@pytest.fixture(autouse=True)
def _restore_modules():
    original = {name: module for name, module in sys.modules.items() if name.startswith(('vlmeval', 'torch'))}
    yield
    for name in list(sys.modules):
        if name.startswith(('vlmeval', 'torch')):
            del sys.modules[name]
    sys.modules.update(original)


def _load_vlm():
    root = Path(__file__).resolve().parents[1]
    for name in ('vlmeval', 'vlmeval.smp'):
        package = types.ModuleType(name)
        package.__path__ = [str(root / name.replace('.', '/'))]
        sys.modules[name] = package
    module = None
    for name in ('smp.log', 'smp.vlm'):
        full_name = 'vlmeval.' + name
        spec = importlib.util.spec_from_file_location(full_name, root / (full_name.replace('.', '/') + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def vlm(monkeypatch):
    # Keep the assertions on the pixels we wrote. The encoder otherwise
    # upscales anything under VLMEVAL_MIN_IMAGE_EDGE (default 100).
    monkeypatch.setenv('VLMEVAL_MIN_IMAGE_EDGE', '1')
    return _load_vlm()


def _png_bytes(image, **save_kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', **save_kwargs)
    return buffer.getvalue()


def _open_b64(payload):
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def _outline(mode):
    """Black square outline on a transparent background, as in issue 1752."""
    base = Image.new('RGBA', (224, 224), (0, 0, 0, 0))
    ImageDraw.Draw(base).rectangle([56, 56, 168, 168], outline=(0, 0, 0, 255), width=8)
    if mode == 'RGBA':
        return base
    if mode == 'LA':
        return base.convert('LA')
    if mode == 'P':
        palette = Image.new('P', base.size, 0)
        # Colour 0 is transparent black, colour 1 is opaque black.
        palette.putpalette([0, 0, 0, 0, 0, 0] + [0] * 762)
        palette.paste(1, mask=base.getchannel('A'))
        palette.info['transparency'] = 0
        return palette
    raise AssertionError(mode)


def _assert_outline_visible(image):
    assert image.getpixel((10, 10)) == (255, 255, 255)
    assert image.getpixel((112, 60)) == (0, 0, 0)


@pytest.mark.parametrize('mode', ['RGBA', 'LA', 'P'])
def test_transparent_outline_stays_visible_after_decode_and_encode(vlm, mode):
    image = _outline(mode)
    raw = _png_bytes(image, transparency=0) if mode == 'P' else _png_bytes(image)
    payload = base64.b64encode(raw)

    _assert_outline_visible(vlm.decode_base64_to_image(payload))

    opened = Image.open(io.BytesIO(raw))
    _assert_outline_visible(_open_b64(vlm.encode_image_to_base64(opened, fmt='PNG')))
    opened = Image.open(io.BytesIO(raw))
    _assert_outline_visible(_open_b64(vlm.encode_image_to_base64(opened)))


def test_partial_alpha_blends_onto_white(vlm):
    image = Image.new('RGBA', (2, 2), (0, 0, 0, 0))
    image.putpixel((0, 0), (0, 0, 0, 128))
    image.putpixel((1, 0), (0, 0, 0, 1))
    image.putpixel((0, 1), (255, 0, 0, 128))
    image.putpixel((1, 1), (10, 20, 30, 200))
    payload = base64.b64encode(_png_bytes(image))

    decoded = vlm.decode_base64_to_image(payload)
    assert decoded.getpixel((0, 0)) == (127, 127, 127)
    assert decoded.getpixel((1, 0)) == (254, 254, 254)
    assert decoded.getpixel((0, 1)) == (255, 127, 127)
    assert decoded.getpixel((1, 1)) == (63, 71, 79)

    encoded = _open_b64(vlm.encode_image_to_base64(image.copy(), fmt='PNG'))
    assert encoded.getpixel((0, 0)) == (127, 127, 127)
    assert encoded.getpixel((1, 1)) == (63, 71, 79)


def test_palette_per_index_alpha_blends_onto_white(vlm):
    image = Image.new('P', (3, 1), 0)
    image.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * 759)
    image.putpixel((0, 0), 0)
    image.putpixel((1, 0), 1)
    image.putpixel((2, 0), 2)
    image.info['transparency'] = bytes([128, 255, 0])

    decoded = vlm.decode_base64_to_image(base64.b64encode(_png_bytes(image)))
    assert decoded.getpixel((0, 0)) == (255, 127, 127)
    assert decoded.getpixel((1, 0)) == (0, 255, 0)
    assert decoded.getpixel((2, 0)) == (255, 255, 255)


def test_color_key_transparency_flattens_onto_white(vlm):
    image = Image.new('RGB', (4, 4), (0, 255, 0))
    image.putpixel((1, 1), (0, 0, 0))
    raw = _png_bytes(image, transparency=(0, 255, 0))
    decoded = vlm.decode_base64_to_image(base64.b64encode(raw))
    assert decoded.getpixel((0, 0)) == (255, 255, 255)
    assert decoded.getpixel((1, 1)) == (0, 0, 0)


@pytest.mark.parametrize('mode', ['PA', 'RGBa'])
def test_similar_alpha_modes_flatten_onto_white(vlm, mode):
    image = Image.new('RGBA', (4, 4), (0, 0, 0, 0))
    image.putpixel((1, 1), (0, 0, 0, 255))
    image.putpixel((2, 2), (255, 0, 0, 128))
    image = image.convert(mode)

    encoded = _open_b64(vlm.encode_image_to_base64(image, fmt='PNG'))
    assert encoded.getpixel((0, 0)) == (255, 255, 255)
    assert encoded.getpixel((1, 1)) == (0, 0, 0)
    assert encoded.getpixel((2, 2)) == (255, 127, 127)


def test_dump_image_path_writes_white_background(vlm, tmp_path):
    raw = _png_bytes(_outline('RGBA'))
    path = tmp_path / 'diagram.png'
    vlm.decode_base64_to_image_file(base64.b64encode(raw), str(path))
    with Image.open(path) as saved:
        _assert_outline_visible(saved.convert('RGB') if saved.mode != 'RGB' else saved)


@pytest.mark.parametrize('mode', ['RGB', 'L'])
def test_opaque_images_without_alpha_roundtrip_unchanged(vlm, mode):
    color = (12, 34, 56) if mode == 'RGB' else 127
    image = Image.new(mode, (32, 32), color)
    image.putpixel((3, 4), (200, 10, 10) if mode == 'RGB' else 200)

    decoded = vlm.decode_base64_to_image(base64.b64encode(_png_bytes(image)))
    assert decoded.mode == mode
    assert decoded.getpixel((0, 0)) == color
    assert decoded.getpixel((3, 4)) == image.getpixel((3, 4))

    for fmt in ('PNG', 'JPEG'):
        direct = io.BytesIO()
        image.save(direct, format=fmt)
        expected = base64.b64encode(direct.getvalue()).decode('utf-8')
        assert vlm.encode_image_to_base64(image.copy(), fmt=fmt) == expected


def test_opaque_i16_decode_keeps_mode_and_pixels(vlm):
    image = Image.new('I;16', (4, 4), 1000)
    image.putpixel((1, 2), 40000)
    decoded = vlm.decode_base64_to_image(base64.b64encode(_png_bytes(image)))
    assert decoded.mode == 'I;16'
    assert decoded.getpixel((0, 0)) == 1000
    assert decoded.getpixel((1, 2)) == 40000


@pytest.mark.parametrize('mode', ['RGBA', 'LA', 'P'])
def test_fully_opaque_images_match_plain_rgb_conversion(vlm, mode):
    if mode == 'RGBA':
        image = Image.new('RGBA', (32, 32), (12, 34, 56, 255))
        image.putpixel((3, 4), (200, 10, 10, 255))
    elif mode == 'LA':
        image = Image.new('LA', (32, 32), (40, 255))
        image.putpixel((3, 4), (200, 255))
    else:
        image = Image.new('P', (32, 32), 1)
        image.putpalette([0, 0, 0, 10, 20, 30] + [0] * 762)
        image.putpixel((3, 4), 1)
    reference = image.convert('RGB')

    decoded = vlm.decode_base64_to_image(base64.b64encode(_png_bytes(image)))
    assert decoded.mode == 'RGB'
    assert decoded.getpixel((0, 0)) == reference.getpixel((0, 0))
    assert decoded.getpixel((3, 4)) == reference.getpixel((3, 4))

    for fmt in ('PNG', 'JPEG'):
        direct = io.BytesIO()
        reference.save(direct, format=fmt)
        expected = base64.b64encode(direct.getvalue()).decode('utf-8')
        assert vlm.encode_image_to_base64(image.copy(), fmt=fmt) == expected


def test_opaque_jpeg_decode_is_unchanged(vlm):
    image = Image.new('RGB', (16, 16), (10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    payload = base64.b64encode(buffer.getvalue())
    decoded = vlm.decode_base64_to_image(payload)
    buffer.seek(0)
    reference = Image.open(buffer)
    assert decoded.mode == reference.mode
    assert decoded.getpixel((0, 0)) == reference.getpixel((0, 0))
