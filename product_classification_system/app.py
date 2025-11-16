#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品自動分類システム - 単一機能版
階層型機械学習による商品分類＋精度検証機能
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import logging
from datetime import datetime

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flaskアプリケーション初期化
app = Flask(__name__)
app.config['SECRET_KEY'] = 'product-classification-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB制限

# テンプレートフォルダの設定
app.template_folder = 'templates'
app.static_folder = 'static'

# スクリプトの場所を基準にした絶対パスを使用
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 必要なフォルダ作成
os.makedirs(os.path.join(SCRIPT_DIR, 'templates'), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, 'static'), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, 'temp_uploads'), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, 'temp_results'), exist_ok=True)

# メインページ
@app.route('/')
def index():
    """商品分類メインページ"""
    return render_template('index.html')

# 商品分類API - 列情報取得
@app.route('/api/columns', methods=['POST'])
def get_columns():
    """市場データの列情報を取得"""
    try:
        from product_classification_web import get_file_columns
        return get_file_columns()
    except ImportError as e:
        logger.error(f"モジュールインポートエラー: {str(e)}")
        return jsonify({
            'error': True, 
            'message': 'product_classification_web.pyが見つかりません'
        }), 500
    except Exception as e:
        logger.error(f"列情報取得エラー: {str(e)}")
        return jsonify({
            'error': True, 
            'message': f'列情報の取得中にエラーが発生しました: {str(e)}'
        }), 500

# 商品分類API - 処理実行
@app.route('/api/process', methods=['POST'])
def process():
    """商品分類処理API"""
    try:
        from product_classification_web import process_product_classification
        return process_product_classification()
    except ImportError as e:
        logger.error(f"モジュールインポートエラー: {str(e)}")
        return jsonify({
            'error': True, 
            'message': 'product_classification_web.pyが見つかりません'
        }), 500
    except Exception as e:
        logger.error(f"商品分類処理エラー: {str(e)}")
        return jsonify({
            'error': True, 
            'message': f'処理中にエラーが発生しました: {str(e)}'
        }), 500

# 商品分類API - ダウンロード
@app.route('/api/download/<download_id>')
def download(download_id):
    """商品分類結果ダウンロード"""
    try:
        from product_classification_web import download_classification_results
        return download_classification_results(download_id)
    except ImportError as e:
        logger.error(f"モジュールインポートエラー: {str(e)}")
        return jsonify({
            'error': True, 
            'message': 'product_classification_web.pyが見つかりません'
        }), 500
    except Exception as e:
        logger.error(f"ダウンロードエラー: {str(e)}")
        return jsonify({
            'error': True, 
            'message': f'ダウンロード中にエラーが発生しました: {str(e)}'
        }), 500

# 商品分類API - クリーンアップ
@app.route('/api/cleanup/<download_id>', methods=['DELETE'])
def cleanup(download_id):
    """一時ファイルクリーンアップ"""
    try:
        from product_classification_web import cleanup_classification_results
        return cleanup_classification_results(download_id)
    except ImportError as e:
        logger.error(f"モジュールインポートエラー: {str(e)}")
        return jsonify({
            'error': True, 
            'message': 'product_classification_web.pyが見つかりません'
        }), 500
    except Exception as e:
        logger.error(f"クリーンアップエラー: {str(e)}")
        return jsonify({
            'error': True, 
            'message': f'クリーンアップ中にエラーが発生しました: {str(e)}'
        }), 500

# ヘルスチェック
@app.route('/api/health')
def health_check():
    """システムヘルスチェック"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'service': 'product_classification'
    })

# エラーハンドラー
@app.errorhandler(404)
def not_found_error(error):
    """404エラーハンドラー"""
    return jsonify({'error': True, 'message': 'ページが見つかりません'}), 404

@app.errorhandler(500)
def internal_error(error):
    """500エラーハンドラー"""
    logger.error(f"内部エラー: {error}")
    return jsonify({'error': True, 'message': 'サーバー内部エラーが発生しました'}), 500

@app.errorhandler(413)
def too_large(error):
    """413エラーハンドラー"""
    return jsonify({
        'error': True, 
        'message': 'ファイルサイズが大きすぎます（50MB以下にしてください）'
    }), 413

# メイン実行
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("商品自動分類システム 起動中...")
    logger.info("=" * 60)
    logger.info("機能: 階層型機械学習による商品分類＋精度検証")
    logger.info("ポート: 5000")
    logger.info("URL: http://localhost:5000")
    logger.info("=" * 60)
    
    # 開発モードで起動
    app.run(debug=True, host='0.0.0.0', port=5000)
