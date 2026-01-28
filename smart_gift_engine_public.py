"""
🎁 Smart Gift Basket Recommendation Engine - Fixed & Perfected
גרסה סופית עם כל התיקונים
"""

import io
import base64
import requests
import csv
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from collections import deque
import math
from itertools import combinations
from typing import List, Dict, Tuple

app = Flask(__name__)
CORS(app)

PIXELS_PER_CM = 12  # הקטן מ-15 ל-12 = מוצרים קטנים ב-20%
MARGIN_MULTIPLIER = 1.3
DEFAULT_SPREADSHEET_ID = "1H_kbTq9-yGBYt3DD7yYLUpT-PnnnJLR6AVgJ3IRJ_V0"
SHEET_NAME = "CLOD"

def get_products_from_public_sheet(spreadsheet_id, sheet_name="CLOD"):
    """קורא מוצרים מ-Google Sheets ציבורי"""
    csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    
    print(f"📊 קורא מהשיטס...")
    
    try:
        response = requests.get(csv_url, timeout=10)
        response.raise_for_status()
        
        csv_data = response.content.decode('utf-8')
        reader = csv.DictReader(io.StringIO(csv_data))
        
        products = []
        for row in reader:
            try:
                if not row.get('product_id') or not row.get('name'):
                    continue
                
                product = {
                    'id': row.get('product_id', ''),
                    'name': row.get('name', ''),
                    'image_url': row.get('image_url', ''),
                    'price': float(row.get('price', 0)),
                    'height_cm': float(row.get('height_cm', 20)),
                    'category': row.get('category', ''),
                    'subcategory': row.get('subcategory', ''),
                    'brand': row.get('brand', ''),
                    'occasions': [x.strip() for x in row.get('occasions', '').split(',') if x.strip()],
                    'recipients': [x.strip() for x in row.get('recipients', '').split(',') if x.strip()],
                    'styles': [x.strip() for x in row.get('styles', '').split(',') if x.strip()],
                    'dietary': [x.strip() for x in row.get('dietary', '').split(',') if x.strip()],
                    'description': row.get('description', ''),
                    'color': row.get('color_dominant', 'white'),
                    'size_category': row.get('size_category', 'medium'),
                    'popularity': int(row.get('popularity_score', 5)) if row.get('popularity_score') else 5,
                    'in_stock': row.get('in_stock', 'TRUE').upper() == 'TRUE'
                }
                
                if product['in_stock']:
                    products.append(product)
                    
            except Exception as e:
                print(f"⚠️ שגיאה בשורה: {e}")
                continue
        
        print(f"✅ נטענו {len(products)} מוצרים")
        return products
        
    except Exception as e:
        print(f"❌ שגיאה: {e}")
        raise Exception(f"לא ניתן לקרוא מהשיטס: {str(e)}")

def calculate_match_score(product: Dict, criteria: Dict) -> Tuple[int, List[str]]:
    """מחשב ציון התאמה"""
    score = 0
    reasons = []
    
    if criteria.get('occasion'):
        if criteria['occasion'] in product['occasions']:
            score += 30
            reasons.append(f"מתאים ל-{criteria['occasion']}")
    
    if criteria.get('recipient_type'):
        if criteria['recipient_type'] in product['recipients']:
            score += 25
            reasons.append(f"מתאים ל-{criteria['recipient_type']}")
    
    if criteria.get('style'):
        if criteria['style'] in product['styles']:
            score += 20
            reasons.append(f"בסגנון {criteria['style']}")
    
    if criteria.get('preferences'):
        prefs = criteria['preferences'] if isinstance(criteria['preferences'], list) else [criteria['preferences']]
        for pref in prefs:
            if pref in product['category'] or pref in product['subcategory']:
                score += 15
                reasons.append(f"כולל {pref}")
                break
    
    if criteria.get('dietary'):
        dietary_reqs = criteria['dietary'] if isinstance(criteria['dietary'], list) else [criteria['dietary']]
        matches_all = all(req in product['dietary'] for req in dietary_reqs)
        if matches_all:
            score += 10
            reasons.append("עונה על דרישות תזונה")
    
    score += product.get('popularity', 5)
    
    return score, reasons

def find_best_basket(products: List[Dict], criteria: Dict) -> Dict:
    """מוצא את המארז הטוב ביותר"""
    budget_min = criteria.get('budget_min', 0)
    budget_max = criteria.get('budget_max', 10000)
    size = criteria.get('size', 'medium')
    
    if size == 'small':
        min_items, max_items = 3, 4
    elif size == 'large':
        min_items, max_items = 8, 12
    else:
        min_items, max_items = 5, 7
    
    filtered_products = products
    if criteria.get('dietary'):
        dietary_reqs = criteria['dietary'] if isinstance(criteria['dietary'], list) else [criteria['dietary']]
        filtered_products = [
            p for p in products
            if all(req in p['dietary'] for req in dietary_reqs)
        ]
    
    if not filtered_products:
        filtered_products = products
    
    scored_products = []
    for product in filtered_products:
        score, reasons = calculate_match_score(product, criteria)
        scored_products.append({
            **product,
            'match_score': score,
            'match_reasons': reasons
        })
    
    scored_products.sort(key=lambda x: x['match_score'], reverse=True)
    
    best_basket = None
    best_score = 0
    
    for num_items in range(min_items, max_items + 1):
        top_products = scored_products[:num_items * 3]
        
        for combo in combinations(top_products, num_items):
            total_cost = sum(p['price'] for p in combo)
            final_price = total_cost * MARGIN_MULTIPLIER
            
            if final_price < budget_min or final_price > budget_max:
                continue
            
            combo_score = sum(p['match_score'] for p in combo)
            categories = set(p['category'] for p in combo)
            combo_score += len(categories) * 5
            colors = set(p['color'] for p in combo)
            combo_score += len(colors) * 3
            sizes = set(p['size_category'] for p in combo)
            combo_score += len(sizes) * 2
            budget_utilization = final_price / budget_max
            if 0.8 <= budget_utilization <= 1.0:
                combo_score += 10
            
            if combo_score > best_score:
                best_score = combo_score
                best_basket = {
                    'products': list(combo),
                    'total_cost': total_cost,
                    'final_price': round(final_price, 2),
                    'score': combo_score,
                    'categories': list(categories),
                    'colors': list(colors)
                }
    
    return best_basket

def generate_explanation(basket: Dict, criteria: Dict) -> str:
    """יוצר הסבר מותאם אישית"""
    products = basket['products']
    
    explanation = "🎁 בחרנו מארז מיוחד זה במיוחד עבורך!\n\n"
    
    if criteria.get('occasion'):
        explanation += f"המארז מתאים באופן מושלם ל-{criteria['occasion']}. "
    if criteria.get('recipient_type'):
        explanation += f"נבחר במיוחד עבור {criteria['recipient_type']}. "
    
    explanation += "\n\n📦 המוצרים שבחרנו:\n\n"
    
    for i, product in enumerate(products, 1):
        reasons = product.get('match_reasons', [])
        explanation += f"{i}. {product['name']} (₪{product['price']})\n"
        if reasons:
            explanation += f"   • {', '.join(reasons[:2])}\n"
    
    explanation += f"\n💰 מחיר סופי: ₪{basket['final_price']}\n"
    explanation += f"(סכום מוצרים: ₪{basket['total_cost']} + מרווח 30%)\n\n"
    
    explanation += "✨ למה המארז הזה מושלם:\n"
    explanation += f"• גיוון של {len(basket['categories'])} קטגוריות שונות\n"
    explanation += f"• שילוב צבעים מושלם ({', '.join(basket['colors'])})\n"
    explanation += f"• המחיר מתאים לתקציב (₪{criteria.get('budget_min', 0)}-₪{criteria.get('budget_max', 0)})\n"
    
    if criteria.get('dietary'):
        dietary = criteria['dietary'] if isinstance(criteria['dietary'], list) else [criteria['dietary']]
        explanation += f"• עונה על דרישות תזונה: {', '.join(dietary)}\n"
    
    return explanation

def remove_background_conservative(img):
    """
    הסרת רקע מהקוד שעובד - ללא שינויים!
    """
    img = img.convert("RGBA")
    width, height = img.size
    pixels = img.load()
    
    mask = Image.new('L', (width, height), 255)
    mask_pixels = mask.load()
    
    edge_colors = []
    for x in range(0, width, max(1, width//20)):
        edge_colors.append(pixels[x, 0][:3])
        edge_colors.append(pixels[x, height-1][:3])
    for y in range(0, height, max(1, height//20)):
        edge_colors.append(pixels[0, y][:3])
        edge_colors.append(pixels[width-1, y][:3])
    
    avg_bg = tuple(sum(c[i] for c in edge_colors) // len(edge_colors) for i in range(3))
    
    visited = set()
    queue = deque()
    
    for x in range(width):
        queue.append((x, 0))
        queue.append((x, height-1))
    for y in range(height):
        queue.append((0, y))
        queue.append((width-1, y))
    
    threshold = 45
    
    while queue:
        x, y = queue.popleft()
        
        if (x, y) in visited or x < 0 or x >= width or y < 0 or y >= height:
            continue
            
        visited.add((x, y))
        
        r, g, b = pixels[x, y][:3]
        diff = abs(r - avg_bg[0]) + abs(g - avg_bg[1]) + abs(b - avg_bg[2])
        
        if diff < threshold:
            mask_pixels[x, y] = 0
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    queue.append((nx, ny))
    
    img.putalpha(mask)
    alpha = img.split()[3]
    alpha = alpha.filter(ImageFilter.GaussianBlur(0.5))
    img.putalpha(alpha)
    
    return img

def arrange_center_out(items):
    """סידור מרכז-החוצה"""
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

def create_professional_collage(basket: Dict) -> Image:
    """יצירת קולאז' מושלם"""
    products = basket['products']
    images_data = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    print(f"📥 מוריד {len(products)} תמונות...")
    
    for product in products:
        try:
            url = product['image_url']
            print(f"   🔗 {product['name'][:30]}...")
            
            # טיפול טוב יותר ב-redirects ו-timeouts
            img_resp = requests.get(
                url, 
                headers=headers, 
                timeout=15,  # חזרה ל-15
                allow_redirects=True,
                max_redirects=5,  # הגבלת redirects
                stream=True
            )
            
            if img_resp.status_code == 200:
                try:
                    # קריאת התוכן עם הגבלת גודל
                    content = img_resp.content[:10*1024*1024]  # מקס 10MB
                    
                    # נסה לפתוח
                    img = Image.open(io.BytesIO(content))
                    
                    # ודא שהתמונה תקינה
                    img.verify()
                    
                    # טען מחדש אחרי verify
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
            img_cropped = img_no_bg.crop(bbox)
            
            # לוג: גודל אמיתי לפני resize
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
    
    print(f"🎯 מסדר {len(processed_images)} מוצרים...")
    
    processed_images.sort(key=lambda x: x.height, reverse=True)
    count = len(processed_images)
    
    if count <= 2:
        rows = [processed_images]
    elif count <= 4:
        rows = [processed_images[:2], processed_images[2:]]
    elif count <= 7:
        rows = [processed_images[:2], processed_images[2:5], processed_images[5:]]
    else:
        # 8+ מוצרים: חלק ל-3 שורות
        per_row = math.ceil(count / 3)
        rows = [processed_images[:per_row], processed_images[per_row:per_row*2], processed_images[per_row*2:]]
    
    arranged_rows = [arrange_center_out(row) for row in rows]
    
    # קנבס קטן יותר!
    max_h = processed_images[0].height
    total_w = sum(img.width for img in processed_images)
    canvas_w = int(total_w * 1.2) + 400  # הקטן מ-1.5
    canvas_h = int(max_h * len(arranged_rows) * 1.1) + 300  # הקטן מ-1.3
    
    print(f"🎬 יוצר רקע...")
    
    # רקע עדין יותר
    final_bg = Image.new("RGB", (canvas_w, canvas_h), (252, 252, 253))
    
    # גרדיאנט עדין מאוד
    center_x_light = canvas_w // 2
    center_y_light = canvas_h * 0.35
    max_radius = math.sqrt((canvas_w/2)**2 + (canvas_h)**2)
    
    for y in range(canvas_h):
        for x in range(canvas_w):
            dist = math.sqrt((x - center_x_light)**2 + (y - center_y_light)**2)
            brightness = 1 - (dist / max_radius) * 0.12  # עדין יותר
            base = 252
            val = int(base * brightness)
            final_bg.putpixel((x, y), (val, val, val + 1))
    
    # מרקם מינימלי
    random.seed(42)
    for y in range(0, canvas_h, 3):
        for x in range(0, canvas_w, 3):
            if random.random() < 0.15:
                current = final_bg.getpixel((x, y))
                noise = random.randint(-2, 2)
                new_val = max(0, min(255, current[0] + noise))
                final_bg.putpixel((x, y), (new_val, new_val, new_val))
    
    final_bg = final_bg.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    print(f"💫 מוסיף צללים...")
    
    shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    center_x = canvas_w // 2
    floor_y = canvas_h - 80
    OVERLAP = 0.12  # הגדל חפיפה מ-0.09 ל-0.12
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
        
        current_x = center_x - total_row_w // 2
        row_y_offset = row_idx * 20
        
        for prod in scaled_row:
            py = floor_y - prod.height + row_y_offset
            px = int(current_x)
            
            all_positions.append({'img': prod, 'x': px, 'y': int(py)})
            
            # צל טבעי יותר
            shadow = prod.copy()
            shadow_data = []
            for item in shadow.getdata():
                if len(item) == 4:
                    # צל כהה יותר (40% במקום 25%)
                    shadow_data.append((20, 20, 20, int(item[3] * 0.40)))
                else:
                    shadow_data.append((20, 20, 20, 100))
            shadow.putdata(shadow_data)
            
            # צל גדול יותר ונמוך יותר
            shadow_w = int(prod.width * 1.0)  # רוחב מלא
            shadow_h = int(prod.height * 0.25)
            shadow = shadow.resize((shadow_w, shadow_h), Image.LANCZOS)
            
            # מיקום צל - יותר למטה
            shadow_x = px + 8
            shadow_y = py + prod.height + 15
            shadow_layer.paste(shadow, (shadow_x, shadow_y), shadow)
            
            current_x += prod.width - int(prod.width * OVERLAP)
    
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=25))  # טשטוש חזק יותר
    final_bg.paste(shadow_layer, (0, 0), shadow_layer)
    
    for pos in all_positions:
        final_bg.paste(pos['img'], (pos['x'], pos['y']), pos['img'])
    
    # חיתוך עם מרווח גדול יותר
    temp_alpha = Image.new("L", (canvas_w, canvas_h), 0)
    for pos in all_positions:
        temp_alpha.paste(pos['img'].split()[3], (pos['x'], pos['y']))
    
    bbox = temp_alpha.getbbox()
    if bbox:
        margin = 180  # הגדל מרווח מ-150 ל-180
        crop_box = (
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(canvas_w, bbox[2] + margin),
            min(canvas_h, bbox[3] + margin)
        )
        final_bg = final_bg.crop(crop_box)
    
    print(f"✨ שיפורים...")
    
    # שיפורים עדינים יותר
    final_bg = ImageEnhance.Contrast(final_bg).enhance(1.12)  # פחות
    final_bg = ImageEnhance.Sharpness(final_bg).enhance(1.15)  # פחות
    final_bg = ImageEnhance.Brightness(final_bg).enhance(1.02)  # פחות
    final_bg = ImageEnhance.Color(final_bg).enhance(1.08)  # פחות
    
    # גודל מקסימלי קטן יותר
    if final_bg.width > 1200 or final_bg.height > 1200:
        final_bg.thumbnail((1200, 1200), Image.LANCZOS)
    
    print(f"🎉 מארז מוכן!")
    
    return final_bg

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "service": "Smart Gift Basket Engine - Perfected",
        "version": "5.0"
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/recommend-basket', methods=['POST'])
def recommend_basket():
    """נקודת קצה ראשית"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        spreadsheet_id = data.get('spreadsheet_id', DEFAULT_SPREADSHEET_ID)
        sheet_name = data.get('sheet_name', SHEET_NAME)
        
        print(f"\n{'='*60}")
        print(f"🎁 מתחיל המלצה - גרסה מושלמת")
        print(f"{'='*60}\n")
        
        products = get_products_from_public_sheet(spreadsheet_id, sheet_name)
        
        if not products:
            return jsonify({"error": "לא נמצאו מוצרים"}), 404
        
        criteria = {
            'budget_min': data.get('budget_min', 0),
            'budget_max': data.get('budget_max', 10000),
            'occasion': data.get('occasion'),
            'recipient_type': data.get('recipient_type'),
            'style': data.get('style'),
            'preferences': data.get('preferences', []),
            'dietary': data.get('dietary', []),
            'size': data.get('size', 'medium')
        }
        
        print(f"🔍 מחפש מארז...")
        
        basket = find_best_basket(products, criteria)
        
        if not basket:
            return jsonify({"error": "לא נמצא מארז מתאים"}), 404
        
        print(f"✅ מארז נמצא! {len(basket['products'])} מוצרים\n")
        
        explanation = generate_explanation(basket, criteria)
        
        print(f"🎨 יוצר תמונה...\n")
        collage_image = create_professional_collage(basket)
        
        buf = io.BytesIO()
        collage_image.save(buf, format='PNG', quality=92, optimize=True)
        buf.seek(0)
        
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        image_data_url = f"data:image/png;base64,{image_base64}"
        
        response = {
            "success": True,
            "basket": {
                "products": [
                    {
                        "id": p['id'],
                        "name": p['name'],
                        "price": p['price'],
                        "category": p['category'],
                        "match_reasons": p.get('match_reasons', [])
                    }
                    for p in basket['products']
                ],
                "total_cost": basket['total_cost'],
                "margin": MARGIN_MULTIPLIER,
                "final_price": basket['final_price'],
                "score": basket['score']
            },
            "collage_image": image_data_url,
            "explanation": explanation,
            "metadata": {
                "total_products_searched": len(products),
                "categories_in_basket": basket['categories'],
                "colors_in_basket": basket['colors'],
                "processing_time": "⚡ מהיר!"
            }
        }
        
        print(f"\n{'='*60}")
        print(f"✅ הצלחה!")
        print(f"{'='*60}\n")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"\n❌ שגיאה: {str(e)}\n")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
