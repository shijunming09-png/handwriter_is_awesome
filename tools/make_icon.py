from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
image = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((7, 7, 249, 249), radius=55, fill='#163b67')
draw.rounded_rectangle((58, 43, 187, 216), radius=13, fill='#f9fbff')
for y, end in ((84, 151), (112, 151), (140, 128), (168, 117)):
    draw.line((83, y, end, y), fill='#aec8df', width=7)
draw.polygon([(113, 182), (128, 146), (202, 68), (225, 90), (149, 168)], fill='#ecb96e')
draw.polygon([(113, 182), (128, 146), (149, 168)], fill='#264b70')
draw.line((130, 160, 213, 78), fill='#fff2d8', width=4)
assets = root / 'assets'
assets.mkdir(exist_ok=True)
image.save(assets / 'handwriter.ico', sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])
image.save(assets / 'handwriter.png')
