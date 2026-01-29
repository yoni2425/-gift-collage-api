"""
🎨 יוצר קולאז' מקצועי
Professional Collage Creator
"""

import io
import math
import random
import requests
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict, Tuple
from background_removal import remove_background_conservative, crop_to_content

# ============================================================
# הגדרות
# ============================================================

PIXELS_PER_CM = 12  # מוצרים קטנים ב-20%
MARGIN_MULTIPLIER = 1.3

# ============================================================
# פונקציות עזר
# ============================================================

def arrange_center_out(items: List[Image]) -> List[Image]:
    """
    מסדר פריטים ממרכז החוצה - הגבוה באמצע
    """
    if not items:
        return []
    
    items_sorted = sorted(items, key=lambda x: x.height, reverse=True)
    center, left, right = [], [], []
    
    for i, item in enumerate(items_sorted):
        if i == 0:
            center.append(item)
        elif i % 2 != 0:
            left.append(item)
        else:
            right.append(item)
    
    left.reverse()
    return left + center + right


def download_and_process_images(products: List[Dict]) -> List[Tuple]:
    """
    מוריד ומעבד תמונות מוצרים
    
    Returns:
        List of (processed_image, height_cm, name)
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    images_data = []
    
    print(f"📥 מוריד {len(products)} תמונות...")
    
    for product in products:
        try:
            url = product['image_url']
            print(f"   🔗 {product['name'][:30]}...")
            
            img_resp = requests.get(
                url, 
                headers=headers, 
                timeout=15,
                allow_redirects=True,
                stream=True
            )
            
            if img_resp.status_code == 200:
                try:
                    # קרא תוכן
                    content = img_resp.content[:10*1024*1024]  # מקס 10MB
                    
                    # פתח תמונה
                    img = Image.open(io.BytesIO(content))
                    
                    # ודא תקינות
                    img.verify()
                    
                    # טען מחדש
                    img = Image.open(io.BytesIO(content))
                    
                    images_data.append((img, product['height_cm'], product['name']))
                    print(f"      ✅")
                    
                except Exception as img_error:
                    print(f"      ❌ תמונה: {str(img_error)[:35]}")
                    continue
            else:
                print(f"      ❌ HTTP {img_resp.status_code}")
                
        except requests.exceptions.TooManyRedirects:
            print(f"      ❌ redirects")
            continue
        except requests.exceptions.Timeout:
            print(f"      ❌ timeout")
            continue
        except Exception as e:
            print(f"      ❌ {str(e)[:35]}")
            continue
    
    if not images_data:
        raise ValueError("לא הצלחתי להוריד תמונות")
    
    return images_data


def process_images(images_data: List[Tuple]) -> List[Image]:
    """
    מעבד תמונות - מסיר רקע וקובע גודל
    """
    print(f"🎨 מעבד {len(images_data)} תמונות...")
    
    processed_images = []
    for img, height_cm, name in images_data:
        try:
            print(f"   מעבד: {name[:25]}...")
            
            # הסר רקע
            img_no_bg = remove_background_conservative(img)
            
            # בדוק מה נשאר
            alpha = img_no_bg.getchannel('A')
            bbox = alpha.getbbox()
            
            if not bbox:
                print(f"      ❌ כל התמונה נמחקה!")
                continue
                
            # חתוך
            img_cropped = crop_to_content(img_no_bg)
            
            # לוג גודל
            real_height = img_cropped.height
            real_width = img_cropped.width
            
            # Resize לפי height_cm
            target_height_px = int(height_cm * PIXELS_PER_CM)
            aspect = real_width / real_height
            target_width_px = int(target_height_px * aspect)
            img_resized = img_cropped.resize((target_width_px, target_height_px), Image.LANCZOS)
            
            processed_images.append(img_resized)
            print(f"      ✅ {real_width}x{real_height} → {target_width_px}x{target_height_px}")
            
        except Exception as e:
            print(f"      ❌ {str(e)[:35]}")
            continue
    
    if not processed_images:
        raise ValueError("לא עובדו תמונות")
    
    return processed_images


def arrange_products(processed_images: List[Image]) -> List[List[Image]]:
    """
    מסדר מוצרים בשורות - טרפז הופכי
    """
    print(f"🎯 מסדר {len(processed_images)} מוצרים...")
    
    processed_images.sort(key=lambda x: x.height, reverse=True)
    count = len(processed_images)
    
    # סידור בטרפז הופכי - רחב מקדימה, תמיד ממורכז!
    if count <= 3:
        # 1-3 מוצרים: שורה אחת
        rows = [processed_images]
    elif count == 4:
        # 4 מוצרים: 1 מאחור, 3 מקדימה
        rows = [processed_images[:1], processed_images[1:]]
    elif count <= 6:
        # 5-6: 2 מאחור, 3-4 מקדימה
        rows = [processed_images[:2], processed_images[2:]]
    elif count <= 9:
        # 7-9: שורות מדורגות
        if count == 7:
            rows = [processed_images[:2], processed_images[2:4], processed_images[4:]]  # 2-2-3
        elif count == 8:
            rows = [processed_images[:2], processed_images[2:5], processed_images[5:]]  # 2-3-3
        else:  # 9
            rows = [processed_images[:3], processed_images[3:6], processed_images[6:]]  # 3-3-3
    elif count <= 12:
        # 10-12: 3 מאחור, 4 אמצע, שאר מקדימה
        rows = [processed_images[:3], processed_images[3:7], processed_images[7:]]
    else:
        # 13+: 3 מאחור, 5 אמצע, שאר מקדימה
        rows = [processed_images[:3], processed_images[3:8], processed_images[8:]]
    
    # סידור כל שורה ממרכז החוצה
    arranged_rows = [arrange_center_out(row) for row in rows]
    
    return arranged_rows


def create_studio_background(canvas_w: int, canvas_h: int) -> Image:
    """
    יוצר רקע בז' חם עם gradient מקצועי
    """
    print(f"🎬 יוצר רקע...")
    
    # רקע בז' חם
    base_color_r = 235
    base_color_g = 225
    base_color_b = 210
    
    final_bg = Image.new("RGB", (canvas_w, canvas_h), (base_color_r, base_color_g, base_color_b))
    
    # גרדיאנט מהצד
    center_x_light = canvas_w * 0.4
    center_y_light = canvas_h * 0.25
    max_radius = math.sqrt((canvas_w)**2 + (canvas_h)**2)
    
    for y in range(canvas_h):
        for x in range(canvas_w):
            dist = math.sqrt((x - center_x_light)**2 + (y - center_y_light)**2)
            brightness = 1 - (dist / max_radius) * 0.15
            
            r = int(base_color_r * brightness)
            g = int(base_color_g * brightness)
            b = int(base_color_b * brightness)
            
            final_bg.putpixel((x, y), (r, g, b))
    
    # מרקם עדין
    random.seed(42)
    for y in range(0, canvas_h, 4):
        for x in range(0, canvas_w, 4):
            if random.random() < 0.12:
                current = final_bg.getpixel((x, y))
                noise = random.randint(-2, 2)
                new_r = max(0, min(255, current[0] + noise))
                new_g = max(0, min(255, current[1] + noise))
                new_b = max(0, min(255, current[2] + noise))
                final_bg.putpixel((x, y), (new_r, new_g, new_b))
    
    # Blur קל
    final_bg = final_bg.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    return final_bg


def render_products_with_shadows(
    arranged_rows: List[List[Image]],
    canvas_w: int,
    canvas_h: int,
    final_bg: Image
) -> Image:
    """
    מציג מוצרים עם צללים על הרקע
    """
    print(f"💫 מוסיף צללים...")
    
    shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    center_x = canvas_w // 2
    floor_y = canvas_h * 0.75
    
    OVERLAP = 0.12
    DEPTH = 0.88
    
    all_positions = []
    
    for row_idx, row_items in enumerate(arranged_rows):
        depth_factor = DEPTH ** (len(arranged_rows) - row_idx - 1)
        
        scaled_row = []
        for img in row_items:
            new_w = int(img.width * depth_factor)
            new_h = int(img.height * depth_factor)
            scaled_row.append(img.resize((new_w, new_h), Image.LANCZOS))
        
        total_row_w = sum(p.width for p in scaled_row)
        if len(scaled_row) > 1:
            total_row_w -= int(scaled_row[0].width * OVERLAP) * (len(scaled_row) - 1)
        
        # בדיקה חכמה: האם צריך להזיז לצד?
        x_offset = 0
        if row_idx > 0 and len(arranged_rows) > 1:
            prev_row = arranged_rows[row_idx - 1]
            if prev_row and scaled_row:
                current_max_h = max(p.height for p in scaled_row)
                prev_max_h = max(p.height for p in prev_row) * depth_factor
                
                coverage_ratio = current_max_h / prev_max_h if prev_max_h > 0 else 0
                
                if coverage_ratio > 0.6:
                    x_offset = 40
                    print(f"      ⚠️ שורה {row_idx + 1} גבוהה ({coverage_ratio:.0%}) - הזזה לצד!")
        
        current_x = center_x - total_row_w // 2 + x_offset
        row_y_offset = row_idx * 25
        
        for prod in scaled_row:
            py = floor_y - prod.height + row_y_offset
            px = int(current_x)
            
            all_positions.append({'img': prod, 'x': px, 'y': int(py)})
            
            # צל כהה וחד
            shadow = prod.copy()
            shadow_data = []
            for item in shadow.getdata():
                if len(item) == 4:
                    shadow_data.append((10, 10, 10, int(item[3] * 0.60)))
                else:
                    shadow_data.append((10, 10, 10, 150))
            shadow.putdata(shadow_data)
            
            shadow_w = int(prod.width * 1.1)
            shadow_h = int(prod.height * 0.22)
            shadow = shadow.resize((shadow_w, shadow_h), Image.LANCZOS)
            
            shadow_x = px + 5
            shadow_y = py + prod.height + 10
            shadow_layer.paste(shadow, (shadow_x, shadow_y), shadow)
            
            current_x += prod.width - int(prod.width * OVERLAP)
    
    # הוסף צללים
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=18))
    final_bg.paste(shadow_layer, (0, 0), shadow_layer)
    
    # הוסף מוצרים
    for pos in all_positions:
        final_bg.paste(pos['img'], (pos['x'], pos['y']), pos['img'])
    
    return final_bg, all_positions


def create_professional_collage(basket: Dict) -> Image:
    """
    פונקציה ראשית - יוצרת קולאז' מקצועי
    
    Args:
        basket: מארז מ-recommendation_engine
        
    Returns:
        תמונת PIL של הקולאז'
    """
    products = basket['products']
    
    if not products:
        raise ValueError("אין מוצרים במארז")
    
    # 1. הורד ועבד תמונות
    images_data = download_and_process_images(products)
    processed_images = process_images(images_data)
    
    # 2. סדר
    arranged_rows = arrange_products(processed_images)
    
    # 3. חשב גודל קנבס
    max_h = processed_images[0].height
    total_w = sum(img.width for img in processed_images)
    canvas_w = int(total_w * 1.2) + 400
    canvas_h = int(max_h * len(arranged_rows) * 1.1) + 300
    
    # 4. צור רקע
    final_bg = create_studio_background(canvas_w, canvas_h)
    
    # 5. הצג מוצרים
    final_bg, all_positions = render_products_with_shadows(arranged_rows, canvas_w, canvas_h, final_bg)
    
    # 6. שיפורים
    print(f"✨ שיפורים...")
    enhancer = ImageEnhance.Contrast(final_bg)
    final_bg = enhancer.enhance(1.12)
    
    enhancer = ImageEnhance.Sharpness(final_bg)
    final_bg = enhancer.enhance(1.15)
    
    enhancer = ImageEnhance.Brightness(final_bg)
    final_bg = enhancer.enhance(1.02)
    
    enhancer = ImageEnhance.Color(final_bg)
    final_bg = enhancer.enhance(1.08)
    
    # 7. חתוך עם שוליים
    temp_alpha = Image.new("L", (canvas_w, canvas_h), 0)
    for pos in all_positions:
        temp_alpha.paste(pos['img'].getchannel('A'), (pos['x'], pos['y']))
    
    bbox = temp_alpha.getbbox()
    if bbox:
        margin = 180
        crop_box = (
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(canvas_w, bbox[2] + margin),
            min(canvas_h, bbox[3] + margin)
        )
        final_bg = final_bg.crop(crop_box)
    
    # 8. הקטן
    if final_bg.width > 1200 or final_bg.height > 1200:
        final_bg.thumbnail((1200, 1200), Image.LANCZOS)
    
    print(f"🎉 מארז מוכן!")
    
    return final_bg
