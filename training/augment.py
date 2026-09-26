from __future__ import annotations
import io,random
from PIL import Image,ImageFilter,ImageEnhance
def jpeg(image): buf=io.BytesIO(); image.save(buf,format="JPEG",quality=random.randint(40,92)); buf.seek(0); return Image.open(buf).convert("RGB")
def mild_blur(image): return image.filter(ImageFilter.GaussianBlur(random.uniform(.25,1.2)))
def exposure(image): return ImageEnhance.Brightness(image).enhance(random.uniform(.85,1.15))
def forensic_augment(image):
    if random.random()<.35: image=jpeg(image)
    if random.random()<.15: image=mild_blur(image)
    if random.random()<.20: image=exposure(image)
    return image
