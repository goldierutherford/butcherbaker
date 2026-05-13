import qrcode
from PIL import Image

try:
    qr = qrcode.QRCode(version=1, box_size=5, border=2)
    qr.add_data("http://127.0.0.1:8000")
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    print(f"QR generated. Mode: {img.mode}, Size: {img.size}")
    img.save("/tmp/test_qr.png")
    print("QR saved to /tmp/test_qr.png")
except Exception as e:
    print(f"Error: {e}")
