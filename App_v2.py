"""
🎁 Gift Basket Recommendation API
Flask Application - Version 2.0 Modular
"""

import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

from recommendation_engine import load_products_from_csv, recommend_basket
from collage_creator import create_professional_collage

app = Flask(__name__)
CORS(app)

# ============================================================
# הגדרות
# ============================================================

DEFAULT_SPREADSHEET_ID = "1H_kbTq9-yGBYt3DD7yYLUpT-PnnnJLR6AVgJ3IRJ_V0"
SHEET_NAME = "CLOD"

# ============================================================
# פונקציות עזר
# ============================================================

def get_products_from_sheets():
    """
    מביא מוצרים מ-Google Sheets
    """
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)
        
        sheet = client.open_by_key(DEFAULT_SPREADSHEET_ID).worksheet(SHEET_NAME)
        data = sheet.get_all_records()
        
        products = []
        for row in data:
            product = {
                'product_id': str(row.get('product_id', '')),
                'name': str(row.get('name', '')),
                'image_url': str(row.get('image_url', '')),
                'price': float(row.get('price', 0)),
                'height_cm': float(row.get('height_cm', 10)),
                'category': str(row.get('category', '')),
                'subcategory': str(row.get('subcategory', '')),
                'brand': str(row.get('brand', '')),
                'occasions': str(row.get('occasions', '')),
                'recipients': str(row.get('recipients', '')),
                'styles': str(row.get('styles', '')),
                'dietary': str(row.get('dietary', '')),
                'description': str(row.get('description', '')),
                'color_dominant': str(row.get('color_dominant', '')),
                'size_category': str(row.get('size_category', 'small')),
                'popularity_score': int(row.get('popularity_score', 5)),
                'in_stock': str(row.get('in_stock', 'TRUE')).upper() == 'TRUE',
                'mishloach_manot_score': int(row.get('mishloach_manot_score', 0)),
                'purim_boost': int(row.get('purim_boost', 0)),
                'chanukah_boost': int(row.get('chanukah_boost', 0)),
                'snack_score': int(row.get('snack_score', 0))
            }
            products.append(product)
        
        return products
        
    except Exception as e:
        print(f"❌ שגיאה בקריאת Sheets: {e}")
        return []

# ============================================================
# Routes
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """בדיקת תקינות"""
    return jsonify({"status": "healthy", "version": "2.0"})


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    ממליץ על מארז ויוצר קולאז'
    
    Body:
    {
        "free_text": "משלוח מנות כיפי עם חטיפים",
        "budget": 150,
        "size": 6  // אופציונלי
    }
    """
    try:
        print("=" * 60)
        print("🎁 מתחיל המלצה - גרסה 2.0 מודולרית")
        print("=" * 60)
        
        data = request.get_json()
        free_text = data.get('free_text', '')
        budget = float(data.get('budget', 200))
        size = data.get('size')
        
        print(f"🔍 מחפש מארז...")
        print(f"   טקסט: {free_text}")
        print(f"   תקציב: ₪{budget}")
        
        # טען מוצרים
        print(f"📊 קורא מהשיטס...")
        products = get_products_from_sheets()
        print(f"✅ נטענו {len(products)} מוצרים")
        
        # המלץ
        criteria = {
            'free_text': free_text,
            'budget': budget,
            'size': size
        }
        
        basket = recommend_basket(products, criteria)
        
        if not basket['products']:
            return jsonify({
                "error": "לא נמצאו מוצרים מתאימים"
            }), 404
        
        print(f"✅ מארז נמצא! {len(basket['products'])} מוצרים")
        
        # צור קולאז'
        print(f"🎨 יוצר תמונה...")
        collage_img = create_professional_collage(basket)
        
        # המר ל-base64
        buffer = io.BytesIO()
        collage_img.save(buffer, format='PNG')
        buffer.seek(0)
        collage_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print("=" * 60)
        print("✅ הצלחה!")
        print("=" * 60)
        
        return jsonify({
            "collage_base64": collage_base64,
            "products": [{
                "id": p['product_id'],
                "name": p['name'],
                "price": p['price'],
                "category": p['category'],
                "match_score": p.get('match_score', 0),
                "match_reasons": p.get('match_reasons', [])
            } for p in basket['products']],
            "total_price": basket['total_price'],
            "budget": basket['budget'],
            "metadata": {
                "total_products_searched": len(products),
                "categories_in_basket": basket['categories'],
                "colors_in_basket": basket['colors'],
                "extracted_tags": basket.get('extracted_tags', {})
            }
        })
        
    except Exception as e:
        print(f"❌ שגיאה: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
