#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品分類システム - Webアプリケーション版
元のTkinterアプリケーションをFlask用に改造 + 精度検証機能追加 + 階層型学習実装
修正: JANコード.0除去対応 + サブセグメント基準の精度検証
"""

from flask import request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
import uuid
import logging
from datetime import datetime
import warnings
import zipfile
import tempfile
import shutil
import re
from collections import defaultdict

warnings.filterwarnings('ignore')

# ログ設定
logger = logging.getLogger(__name__)

# 設定
UPLOAD_FOLDER = 'temp_uploads'
RESULTS_FOLDER = 'temp_results'

# フォルダ作成
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def normalize_jan(jan_value) -> str:
    """JANコードを正規化（.0除去、文字列化、トリミング）"""
    if pd.isna(jan_value) or jan_value == '':
        return ''
    
    jan_str = str(jan_value).strip()
    
    # .0を除去
    if jan_str.endswith('.0'):
        jan_str = jan_str[:-2]
    
    # nanやNoneを除外
    if jan_str.lower() in ['nan', 'none', '']:
        return ''
    
    return jan_str

class ProductClassifierWeb:
    """Webアプリケーション用商品分類エンジン（階層型学習対応）"""
    
    def __init__(self):
        self.vectorizer = None
        self.models = {}
        self.label_encoders = {}
        self.jan_dict = {}
        self.is_trained = False
        # 階層型学習用のマッピング
        self.hierarchy_mapping = {}
        self.use_hierarchical = True  # デフォルトで階層型学習を使用
        
    def normalize_text(self, text: str) -> str:
        """日本語テキストの正規化処理"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        # 全角・半角統一、大文字・小文字統一
        text = text.lower()
        # カタカナ・ひらがなはそのまま（商品名の特徴を保持）
        return text.strip()
    
    def extract_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """数値特徴量を抽出（容量・入数・価格）"""
        result_df = df.copy()
        
        # 価格の数値化
        if 'avg_price' in df.columns:
            def extract_price(price_str):
                if pd.isna(price_str) or price_str == '':
                    return 0
                # 数字部分を抽出
                matches = re.findall(r'[\d,]+', str(price_str))
                if matches:
                    try:
                        return float(matches[0].replace(',', ''))
                    except:
                        return 0
                return 0
            
            result_df['price_numeric'] = df['avg_price'].apply(extract_price)
        
        # 容量の数値化（standardカラムから）
        if 'standard' in df.columns:
            def extract_volume(standard_str):
                if pd.isna(standard_str) or standard_str == '':
                    return 0
                # ml, g, Lなどの容量を抽出
                matches = re.findall(r'(\d+(?:\.\d+)?)\s*(ml|g|kg|l)', str(standard_str).lower())
                if matches:
                    value, unit = matches[0]
                    value = float(value)
                    # グラム換算
                    if unit in ['kg']:
                        value *= 1000
                    elif unit in ['l']:
                        value *= 1000  # mlに変換後、グラム扱い
                    return value
                return 0
            
            def extract_count(standard_str):
                if pd.isna(standard_str) or standard_str == '':
                    return 1
                # 入数を抽出
                matches = re.findall(r'(\d+)\s*個|(\d+)\s*本|(\d+)\s*袋', str(standard_str))
                if matches:
                    for match in matches:
                        for count in match:
                            if count:
                                return int(count)
                return 1
            
            result_df['volume_numeric'] = df['standard'].apply(extract_volume)
            result_df['count_numeric'] = df['standard'].apply(extract_count)
        
        # JANコードからメーカーコード抽出
        if 'jan' in df.columns:
            def extract_maker_code(jan_str):
                if pd.isna(jan_str) or jan_str == '':
                    return ''
                jan_str = str(jan_str).strip()
                if len(jan_str) >= 7:
                    return jan_str[:7]
                return ''
            
            result_df['jan_maker_code'] = df['jan'].apply(extract_maker_code)
        
        return result_df
    
    def create_hierarchy_mapping(self, trial_df: pd.DataFrame):
        """階層構造マッピングを作成"""
        try:
            # カラム名を正規化（前後の空白を削除）
            trial_df.columns = trial_df.columns.str.strip()
            
            # デバッグ用：利用可能なカラム名をログ出力
            logger.info(f"利用可能なカラム名: {list(trial_df.columns)}")
            
            # 必要なカラム名の存在確認
            required_columns = ['カテゴリー名', 'サブカテゴリー名', 'セグメント名', 'サブセグメント名']
            missing_columns = [col for col in required_columns if col not in trial_df.columns]
            
            if missing_columns:
                logger.error(f"必要なカラムが見つかりません: {missing_columns}")
                raise ValueError(f"必要なカラムが見つかりません: {missing_columns}")
            
            self.hierarchy_mapping = {
                'category_to_subcategory': defaultdict(set),
                'subcategory_to_segment': defaultdict(set),
                'segment_to_subsegment': defaultdict(set)
            }
            
            for _, row in trial_df.iterrows():
                category = row['カテゴリー名']
                subcategory = row['サブカテゴリー名']
                segment = row['セグメント名']
                subsegment = row['サブセグメント名']
                
                # 階層関係を記録
                if pd.notna(category) and pd.notna(subcategory):
                    self.hierarchy_mapping['category_to_subcategory'][category].add(subcategory)
                if pd.notna(subcategory) and pd.notna(segment):
                    self.hierarchy_mapping['subcategory_to_segment'][subcategory].add(segment)
                if pd.notna(segment) and pd.notna(subsegment):
                    self.hierarchy_mapping['segment_to_subsegment'][segment].add(subsegment)
            
            # setをlistに変換
            for key in self.hierarchy_mapping:
                for parent_key in self.hierarchy_mapping[key]:
                    self.hierarchy_mapping[key][parent_key] = list(self.hierarchy_mapping[key][parent_key])
            
            logger.info(f"階層マッピング作成完了: "
                       f"カテゴリー-サブカテゴリー: {len(self.hierarchy_mapping['category_to_subcategory'])}件, "
                       f"サブカテゴリー-セグメント: {len(self.hierarchy_mapping['subcategory_to_segment'])}件, "
                       f"セグメント-サブセグメント: {len(self.hierarchy_mapping['segment_to_subsegment'])}件")
            
        except Exception as e:
            logger.error(f"階層マッピング作成エラー: {e}")
            raise
    
    def get_valid_choices_for_hierarchy(self, parent_prediction: str, hierarchy_level: str) -> list:
        """指定された上位階層に基づいて有効な選択肢を取得"""
        mapping_key = {
            'subcategory': 'category_to_subcategory',
            'segment': 'subcategory_to_segment',
            'subsegment': 'segment_to_subsegment'
        }.get(hierarchy_level)
        
        if mapping_key and parent_prediction in self.hierarchy_mapping[mapping_key]:
            return self.hierarchy_mapping[mapping_key][parent_prediction]
        return []
    
    def find_closest_valid_choice(self, prediction: str, valid_choices: list, 
                                 label_encoder: LabelEncoder) -> tuple:
        """制約に合わない予測結果を最も近い有効な選択肢に修正"""
        if not valid_choices or prediction in valid_choices:
            return prediction, 1.0
        
        try:
            # 文字列の類似度計算（簡単なマッチング）
            best_choice = valid_choices[0]
            best_score = 0
            
            for choice in valid_choices:
                # 共通文字数による類似度計算
                common_chars = set(prediction) & set(choice)
                similarity = len(common_chars) / max(len(prediction), len(choice), 1)
                
                if similarity > best_score:
                    best_score = similarity
                    best_choice = choice
            
            # 信頼度を下げる（類似度に応じて調整）
            confidence_penalty = max(0.3, best_score)
            
            return best_choice, confidence_penalty
            
        except Exception as e:
            logger.warning(f"最適選択肢検索エラー: {e}")
            return valid_choices[0] if valid_choices else prediction, 0.3

    def prepare_features(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """特徴量を作成（文字n-gram + 数値特徴量）"""
        try:
            # テキスト特徴量の正規化と結合
            text_features = []
            for _, row in df.iterrows():
                text_parts = []
                for col in feature_cols:
                    if col in ['price_numeric', 'volume_numeric', 'count_numeric']:
                        continue  # 数値特徴は別途処理
                    text_parts.append(self.normalize_text(row.get(col, '')))
                text_features.append(' '.join(text_parts))
            
            if self.vectorizer is None:
                # 文字n-gramベースのTF-IDFベクトライザー
                self.vectorizer = TfidfVectorizer(
                    analyzer='char',
                    ngram_range=(2, 4),
                    max_features=2000,
                    min_df=2,
                    max_df=0.9
                )
                text_matrix = self.vectorizer.fit_transform(text_features).toarray()
            else:
                text_matrix = self.vectorizer.transform(text_features).toarray()
            
            # 数値特徴量を追加（もしあれば）
            numeric_features = []
            for col in ['price_numeric', 'volume_numeric', 'count_numeric']:
                if col in df.columns:
                    # 正規化（0-1スケール）
                    values = df[col].fillna(0).values
                    if np.max(values) > 0:
                        values = values / np.max(values)
                    numeric_features.append(values.reshape(-1, 1))
            
            if numeric_features:
                numeric_matrix = np.hstack(numeric_features)
                # テキスト特徴量と数値特徴量を結合
                features = np.hstack([text_matrix, numeric_matrix])
            else:
                features = text_matrix
            
            return features
        except Exception as e:
            logger.error(f"特徴量作成エラー: {e}")
            raise
    
    def create_jan_dict(self, trial_df: pd.DataFrame):
        """JAN辞書を作成"""
        try:
            # カラム名を正規化（前後の空白を削除）
            trial_df.columns = trial_df.columns.str.strip()
            
            self.jan_dict = {}
            for _, row in trial_df.iterrows():
                # 修正: normalize_jan関数を使用
                jan = normalize_jan(row['JAN'])
                
                if jan:
                    self.jan_dict[jan] = {
                        'カテゴリー名': row['カテゴリー名'],
                        'サブカテゴリー名': row['サブカテゴリー名'],
                        'セグメント名': row['セグメント名'],
                        'サブセグメント名': row['サブセグメント名']
                    }
            logger.info(f"JAN辞書作成完了: {len(self.jan_dict)}件")
        except Exception as e:
            logger.error(f"JAN辞書作成エラー: {e}")
            raise

    def validate_accuracy(self, matched_data: pd.DataFrame, feature_cols: list, target_cols: list):
        """精度検証を実行（9段階：10%-90%）階層型学習対応
        修正: サブセグメント基準でクラス数チェック"""
        try:
            train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            validation_results = {
                'train_ratios': train_ratios,
                'accuracy_scores': {col: [] for col in target_cols},
                'sample_counts': []
            }
            
            logger.info("精度検証開始（階層型学習）...")
            
            for i, train_ratio in enumerate(train_ratios):
                logger.info(f"検証 {i+1}/9: 学習データ{int(train_ratio*100)}%")
                
                # データ分割
                n_samples = len(matched_data)
                n_train = int(n_samples * train_ratio)
                
                # ランダムサンプリング
                shuffled_data = matched_data.sample(frac=1, random_state=42+i).reset_index(drop=True)
                train_data = shuffled_data.iloc[:n_train]
                test_data = shuffled_data.iloc[n_train:]
                
                # 修正: サブセグメント（最下層）でクラス数をチェック
                subsegment_col = 'predicted_subsegment'  # 最下層のカラム
                if subsegment_col in train_data.columns:
                    unique_subsegments = train_data[subsegment_col].nunique()
                    if unique_subsegments < 2:
                        logger.warning(f"検証 {i+1}/9 スキップ: サブセグメントが{unique_subsegments}クラスのみ（最低2クラス必要）")
                        validation_results['sample_counts'].append({
                            'train': len(train_data),
                            'test': len(test_data)
                        })
                        for target_col in target_cols:
                            validation_results['accuracy_scores'][target_col].append(0.0)
                        continue
                
                validation_results['sample_counts'].append({
                    'train': len(train_data),
                    'test': len(test_data)
                })
                
                # 一時的なクラシファイア作成
                temp_classifier = ProductClassifierWeb()
                
                # 既存の階層マッピングをコピー（検証時は新しく作成しない）
                temp_classifier.hierarchy_mapping = self.hierarchy_mapping.copy()
                temp_classifier.use_hierarchical = self.use_hierarchical
                
                # 学習実行（階層マッピング作成をスキップ）
                try:
                    temp_classifier._train_without_hierarchy_mapping(train_data, feature_cols, target_cols)
                except Exception as e:
                    logger.error(f"検証 {i+1}/9 訓練エラー: {str(e)}")
                    for target_col in target_cols:
                        validation_results['accuracy_scores'][target_col].append(0.0)
                    continue
                
                # 予測実行（階層型予測使用）
                try:
                    predictions = temp_classifier.predict_batch(test_data, feature_cols, target_cols, 
                                                              use_hierarchical=self.use_hierarchical, batch_size=50)
                except Exception as e:
                    logger.error(f"検証 {i+1}/9 予測エラー: {str(e)}")
                    for target_col in target_cols:
                        validation_results['accuracy_scores'][target_col].append(0.0)
                    continue
                
                # 各分類レベルでの精度計算
                for target_col in target_cols:
                    if target_col in predictions.columns and target_col in test_data.columns:
                        y_true = test_data[target_col]
                        y_pred = predictions[target_col]
                        
                        accuracy = accuracy_score(y_true, y_pred)
                        validation_results['accuracy_scores'][target_col].append(round(accuracy, 4))
                    else:
                        validation_results['accuracy_scores'][target_col].append(0.0)
                
                logger.info(f"検証 {i+1}/9 完了")
            
            logger.info("精度検証完了")
            return validation_results
            
        except Exception as e:
            logger.error(f"精度検証エラー: {e}")
            return None
    
    def train(self, train_df: pd.DataFrame, feature_cols: list, target_cols: list, 
              use_hierarchical: bool = True):
        """モデルを訓練（階層型学習対応）"""
        try:
            # 有効なデータのみ使用
            train_df = train_df.dropna(subset=target_cols)
            if len(train_df) == 0:
                raise ValueError("訓練データが空です")
            
            # 階層型学習の場合、階層マッピングを作成（トライアルマスターから）
            if use_hierarchical and not self.hierarchy_mapping:
                # この時点では階層マッピングは既に作成済みであるべき
                logger.warning("階層マッピングが未作成のため従来型学習に切り替えます")
                use_hierarchical = False
            
            self.use_hierarchical = use_hierarchical
            
            # 実際の学習処理
            self._train_without_hierarchy_mapping(train_df, feature_cols, target_cols)
            
            hierarchical_info = "階層型" if use_hierarchical else "従来型"
            logger.info(f"モデル訓練完了（{hierarchical_info}）: {len(train_df)}件のデータで学習（LinearSVC + 文字n-gram）")
            
        except Exception as e:
            logger.error(f"モデル訓練エラー: {e}")
            raise
    
    def _train_without_hierarchy_mapping(self, train_df: pd.DataFrame, feature_cols: list, target_cols: list):
        """階層マッピング作成を行わないモデル訓練（検証用）"""
        try:
            # 数値特徴量を抽出
            train_df = self.extract_numeric_features(train_df)
            
            # 特徴量作成
            X = self.prepare_features(train_df, feature_cols)
            
            # 各ターゲットに対してモデルを訓練
            for target_col in target_cols:
                # ラベルエンコーディング
                le = LabelEncoder()
                y = le.fit_transform(train_df[target_col].fillna('不明'))
                self.label_encoders[target_col] = le
                
                # LinearSVCモデル（クラス不均衡に対応）
                model = LinearSVC(
                    class_weight='balanced',
                    random_state=42,
                    max_iter=2000,
                    dual=False  # 特徴量数がサンプル数より多い場合のため
                )
                model.fit(X, y)
                self.models[target_col] = model
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"モデル訓練エラー（内部）: {e}")
            raise
    
    def predict_batch(self, df: pd.DataFrame, feature_cols: list, 
                     target_cols: list, use_hierarchical: bool = True, 
                     batch_size: int = 100) -> pd.DataFrame:
        """バッチ予測を実行（階層型学習対応）"""
        try:
            if not self.is_trained:
                raise ValueError("モデルが訓練されていません")
            
            # use_hierarchical フラグがTrueかつ階層マッピングがある場合のみ階層型予測を使用
            use_hierarchical = use_hierarchical and bool(self.hierarchy_mapping)
            
            # 数値特徴量を抽出
            df = self.extract_numeric_features(df)
            
            results = []
            total_rows = len(df)
            
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i+batch_size]
                X = self.prepare_features(batch, feature_cols)
                
                batch_results = batch.copy()
                
                if use_hierarchical:
                    # 階層型予測
                    batch_results = self._hierarchical_predict(batch_results, X, target_cols)
                else:
                    # 従来型予測
                    batch_results = self._standard_predict(batch_results, X, target_cols)
                
                results.append(batch_results)
            
            prediction_type = "階層型" if use_hierarchical else "従来型"
            logger.info(f"{prediction_type}予測完了: {total_rows}件")
            
            return pd.concat(results, ignore_index=True)
            
        except Exception as e:
            logger.error(f"予測エラー: {e}")
            raise
    
    def _standard_predict(self, batch_results: pd.DataFrame, X: np.ndarray, 
                         target_cols: list) -> pd.DataFrame:
        """従来型予測（階層制約なし）"""
        confidences = []
        
        for target_col in target_cols:
            model = self.models[target_col]
            le = self.label_encoders[target_col]
            
            # 予測
            predictions = model.predict(X)
            
            # LinearSVCは決定関数の値を使って信頼度を計算
            decision_scores = model.decision_function(X)
            if decision_scores.ndim > 1:
                # 多クラスの場合、最大スコアを正規化して信頼度とする
                max_scores = np.max(decision_scores, axis=1)
                confidences_normalized = 1 / (1 + np.exp(-max_scores))  # シグモイド変換
            else:
                # 2クラスの場合
                confidences_normalized = 1 / (1 + np.exp(-np.abs(decision_scores)))
            
            # デコード
            decoded_predictions = le.inverse_transform(predictions)
            batch_results[target_col] = decoded_predictions
            
            confidences.append(confidences_normalized)
        
        # 平均信頼度
        batch_results['confidence'] = np.mean(confidences, axis=0)
        
        return batch_results
    
    def _hierarchical_predict(self, batch_results: pd.DataFrame, X: np.ndarray, 
                             target_cols: list) -> pd.DataFrame:
        """階層型予測（階層制約あり）"""
        # 階層順序
        hierarchy_order = [
            ('predicted_category', None),
            ('predicted_subcategory', 'subcategory'),
            ('predicted_segment', 'segment'),
            ('predicted_subsegment', 'subsegment')
        ]
        
        batch_confidences = []
        
        for idx in range(len(batch_results)):
            row_predictions = {}
            row_confidences = []
            
            for target_col, hierarchy_level in hierarchy_order:
                model = self.models[target_col]
                le = self.label_encoders[target_col]
                
                # 単一サンプル予測
                X_single = X[idx:idx+1]
                predictions = model.predict(X_single)
                decision_scores = model.decision_function(X_single)
                
                # 信頼度計算
                if decision_scores.ndim > 1:
                    max_score = np.max(decision_scores)
                    base_confidence = 1 / (1 + np.exp(-max_score))
                else:
                    base_confidence = 1 / (1 + np.exp(-np.abs(decision_scores[0])))
                
                # デコード
                decoded_prediction = le.inverse_transform(predictions)[0]
                
                # 階層制約チェック（カテゴリー以外）
                if hierarchy_level:
                    parent_col = hierarchy_order[hierarchy_order.index((target_col, hierarchy_level)) - 1][0]
                    parent_prediction = row_predictions[parent_col]
                    
                    valid_choices = self.get_valid_choices_for_hierarchy(
                        parent_prediction, hierarchy_level
                    )
                    
                    if valid_choices and decoded_prediction not in valid_choices:
                        # 制約に合わない場合、最も近い有効な選択肢に修正
                        corrected_prediction, penalty = self.find_closest_valid_choice(
                            decoded_prediction, valid_choices, le
                        )
                        decoded_prediction = corrected_prediction
                        base_confidence *= penalty
                
                row_predictions[target_col] = decoded_prediction
                row_confidences.append(base_confidence)
            
            # 結果を格納
            for target_col, _ in hierarchy_order:
                batch_results.loc[batch_results.index[idx], target_col] = row_predictions[target_col]
            
            batch_confidences.append(np.mean(row_confidences))
        
        batch_results['confidence'] = batch_confidences
        
        return batch_results

def get_file_columns():
    """アップロードされたファイルのカラム情報を取得"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': True, 'message': 'ファイルが選択されていません'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': True, 'message': 'ファイルが選択されていません'})
        
        # 一時ファイルに保存
        temp_id = str(uuid.uuid4())
        temp_path = os.path.join(UPLOAD_FOLDER, f"{temp_id}_{file.filename}")
        file.save(temp_path)
        
        try:
            # ファイル読み込み
            if file.filename.endswith('.csv'):
                # CSV文字コード自動判定
                try:
                    df = pd.read_csv(temp_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(temp_path, encoding='shift_jis')
            else:
                df = pd.read_excel(temp_path)
            
            columns = list(df.columns)
            
            return jsonify({
                'error': False,
                'columns': columns
            })
            
        finally:
            # 一時ファイル削除
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"カラム情報取得エラー: {e}")
        return jsonify({'error': True, 'message': str(e)})

def process_product_classification():
    """商品分類処理のメイン関数（階層型学習対応）
    修正: JANコード.0除去対応"""
    try:
        # ファイルチェック
        if 'market_file' not in request.files or 'trial_file' not in request.files:
            return jsonify({'error': True, 'message': '両方のファイルが必要です'})
        
        market_file = request.files['market_file']
        trial_file = request.files['trial_file']
        
        # パラメータ取得
        jan_column = request.form.get('jan_column')
        product_column = request.form.get('product_column')
        standard_column = request.form.get('standard_column', '')
        manufacturer_column = request.form.get('manufacturer_column', '')
        container_column = request.form.get('container_column', '')
        avg_price_column = request.form.get('avg_price_column', '')
        
        # 修正: 大分類、中分類、小分類、細分類のカラム情報を取得
        category_column = request.form.get('category_column', '')
        subcategory_column = request.form.get('subcategory_column', '')
        segment_column = request.form.get('segment_column', '')
        subsegment_column = request.form.get('subsegment_column', '')
        
        if not jan_column or not product_column:
            return jsonify({'error': True, 'message': '必須項目が不足しています'})
        
        # 一意なIDで一時ファイル保存
        process_id = str(uuid.uuid4())
        market_path = os.path.join(UPLOAD_FOLDER, f"{process_id}_market_{market_file.filename}")
        trial_path = os.path.join(UPLOAD_FOLDER, f"{process_id}_trial_{trial_file.filename}")
        
        market_file.save(market_path)
        trial_file.save(trial_path)
        
        logger.info(f"分類処理開始（階層型学習）: {process_id}")
        
        try:
            # データ読み込み
            market_df = load_file(market_path)
            trial_df = load_file(trial_path)
            
            # トライアルマスターの必須カラムチェック
            required_trial_cols = ['JAN', '商品名', '規格', 'メーカー名', 
                                 'カテゴリー名', 'サブカテゴリー名', 'セグメント名', 'サブセグメント名']
            missing_cols = [col for col in required_trial_cols if col not in trial_df.columns]
            if missing_cols:
                return jsonify({'error': True, 'message': f'トライアルマスターに必須カラムが不足: {", ".join(missing_cols)}'})
            
            # 市場データのカラムマッピング
            market_df = market_df.rename(columns={
                jan_column: 'jan',
                product_column: 'product_name'
            })
            
            if standard_column:
                market_df['standard'] = market_df[standard_column]
            else:
                market_df['standard'] = ''
                
            if manufacturer_column:
                market_df['manufacturer'] = market_df[manufacturer_column]
            else:
                market_df['manufacturer'] = ''

            if container_column:
                market_df['container'] = market_df[container_column]
            else:
                market_df['container'] = ''

            if avg_price_column:
                market_df['avg_price'] = market_df[avg_price_column]
            else:
                market_df['avg_price'] = ''
            
            # 修正: 大分類、中分類、小分類、細分類を特徴量として追加
            if category_column:
                market_df['existing_category'] = market_df[category_column]
            else:
                market_df['existing_category'] = ''
                
            if subcategory_column:
                market_df['existing_subcategory'] = market_df[subcategory_column]
            else:
                market_df['existing_subcategory'] = ''
                
            if segment_column:
                market_df['existing_segment'] = market_df[segment_column]
            else:
                market_df['existing_segment'] = ''
                
            if subsegment_column:
                market_df['existing_subsegment'] = market_df[subsegment_column]
            else:
                market_df['existing_subsegment'] = ''
            
            # 修正: JANを文字列に変換（.0除去対応）
            market_df['jan'] = market_df['jan'].apply(normalize_jan)
            trial_df['JAN'] = trial_df['JAN'].apply(normalize_jan)
            
            # 分類器初期化
            classifier = ProductClassifierWeb()
            
            # JAN辞書作成
            classifier.create_jan_dict(trial_df)
            
            # 階層マッピング作成（トライアルマスターから）
            if len(trial_df) > 0:
                classifier.create_hierarchy_mapping(trial_df)
            
            # JAN照合
            jan_matched_list = []
            jan_unmatched_list = []
            
            for _, row in market_df.iterrows():
                # 修正: normalize_jan関数を使用
                jan = normalize_jan(row['jan'])
                
                if jan in classifier.jan_dict:
                    matched_data = classifier.jan_dict[jan]
                    result_row = row.to_dict()
                    result_row.update({
                        'predicted_category': matched_data['カテゴリー名'],
                        'predicted_subcategory': matched_data['サブカテゴリー名'],
                        'predicted_segment': matched_data['セグメント名'],
                        'predicted_subsegment': matched_data['サブセグメント名'],
                        'confidence': 1.0,
                        'status': 'jan_matched',
                        'method': 'JAN照合'
                    })
                    jan_matched_list.append(result_row)
                else:
                    jan_unmatched_list.append(row)
            
            logger.info(f"JAN照合完了: {len(jan_matched_list)}件一致")
            
            # 精度検証実行（JANマッチしたデータで、階層型学習使用）
            validation_results = None
            if len(jan_matched_list) > 50:  # 十分なデータがある場合のみ検証
                logger.info("精度検証開始（階層型学習）...")
                matched_df = pd.DataFrame(jan_matched_list)
                
                # 修正: 利用可能な特徴量を動的に決定（分類カラムを含む）
                available_features = []
                base_features = ['product_name', 'standard', 'manufacturer', 'container', 'avg_price', 'jan_maker_code',
                                'existing_category', 'existing_subcategory', 'existing_segment', 'existing_subsegment']
                
                for col in base_features:
                    if col in matched_df.columns and matched_df[col].notna().any() and (matched_df[col] != '').any():
                        available_features.append(col)
                
                if not available_features:
                    available_features = ['product_name']
                
                target_cols = ['predicted_category', 'predicted_subcategory', 
                             'predicted_segment', 'predicted_subsegment']
                
                validation_results = classifier.validate_accuracy(matched_df, available_features, target_cols)
            
            # 機械学習による予測（階層型学習使用）
            ml_results_list = []
            
            if len(jan_unmatched_list) > 0 and len(jan_matched_list) > 0:
                logger.info("階層型機械学習モデル訓練開始...")
                
                # 訓練データ準備
                train_df = pd.DataFrame(jan_matched_list)
                
                # 修正: 利用可能な特徴量を動的に決定（分類カラムを含む）
                available_features = []
                base_features = ['product_name', 'standard', 'manufacturer', 'container', 'avg_price', 'jan_maker_code',
                                'existing_category', 'existing_subcategory', 'existing_segment', 'existing_subsegment']
                
                for col in base_features:
                    if col in train_df.columns and train_df[col].notna().any() and (train_df[col] != '').any():
                        available_features.append(col)
                
                if not available_features:
                    available_features = ['product_name']
                
                target_cols = ['predicted_category', 'predicted_subcategory',
                             'predicted_segment', 'predicted_subsegment']
                
                # モデル訓練（階層型学習使用）
                classifier.train(train_df, available_features, target_cols, use_hierarchical=True)
                
                # 予測実行
                logger.info(f"階層型ML予測実行中: {len(jan_unmatched_list)}件...")
                
                unmatched_df = pd.DataFrame(jan_unmatched_list)
                
                # 予測用データにも同じ特徴量カラムを確保
                for col in available_features:
                    if col not in unmatched_df.columns:
                        unmatched_df[col] = ''
                
                predictions_df = classifier.predict_batch(
                    unmatched_df, available_features, target_cols, 
                    use_hierarchical=True, batch_size=50
                )
                
                # ステータス設定
                for _, row in predictions_df.iterrows():
                    confidence = row['confidence']
                    if confidence >= 0.7:
                        status = 'ml_high'
                    elif confidence >= 0.4:
                        status = 'ml_medium'
                    else:
                        status = 'ml_low'
                    
                    result_row = row.to_dict()
                    result_row.update({
                        'status': status,
                        'method': '階層型ML予測'
                    })
                    ml_results_list.append(result_row)
                
                logger.info(f"階層型ML予測完了: {len(ml_results_list)}件")
            
            elif len(jan_unmatched_list) > 0:
                # 訓練データがない場合
                logger.warning("警告: JAN一致データがないためMLスキップ")
                for row in jan_unmatched_list:
                    result_row = row.to_dict()
                    result_row.update({
                        'predicted_category': '不明',
                        'predicted_subcategory': '不明',
                        'predicted_segment': '不明',
                        'predicted_subsegment': '不明',
                        'confidence': 0.0,
                        'status': 'ml_low',
                        'method': 'データ不足'
                    })
                    ml_results_list.append(result_row)
            
            # 結果統合
            all_results = jan_matched_list + ml_results_list
            
            if not all_results:
                return jsonify({'error': True, 'message': '処理結果が空です'})
            
            results_df = pd.DataFrame(all_results)
            
            # 結果保存
            result_filename = f"商品分類結果_階層型_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            result_path = os.path.join(RESULTS_FOLDER, f"{process_id}_{result_filename}")
            
            # Excel出力用にカラム順序を整理
            output_columns = [
                'jan', 'product_name', 'standard', 'manufacturer',
                'predicted_category', 'predicted_subcategory', 'predicted_segment', 'predicted_subsegment',
                'confidence', 'status', 'method'
            ]
            
            # 存在するカラムのみ選択
            existing_columns = [col for col in output_columns if col in results_df.columns]
            output_df = results_df[existing_columns].copy()
            
            # カラム名を日本語に変更
            column_rename = {
                'jan': 'JAN',
                'product_name': '商品名',
                'standard': '規格',
                'manufacturer': 'メーカー',
                'predicted_category': '予測カテゴリー',
                'predicted_subcategory': '予測サブカテゴリー',
                'predicted_segment': '予測セグメント',
                'predicted_subsegment': '予測サブセグメント',
                'confidence': '信頼度',
                'status': 'ステータス',
                'method': '手法'
            }
            
            output_df = output_df.rename(columns=column_rename)
            
            # Excel保存
            output_df.to_excel(result_path, index=False)
            
            # 統計情報
            jan_matched_count = len([r for r in all_results if r['status'] == 'jan_matched'])
            ml_high_count = len([r for r in all_results if r['status'] == 'ml_high'])
            ml_medium_count = len([r for r in all_results if r['status'] == 'ml_medium'])
            ml_low_count = len([r for r in all_results if r['status'] == 'ml_low'])
            
            statistics = {
                'total': len(all_results),
                'jan_matched': jan_matched_count,
                'ml_high': ml_high_count,
                'ml_medium': ml_medium_count,
                'ml_low': ml_low_count
            }
            
            # 全結果をフロントエンドに送信
            formatted_results = []
            for result in all_results:
                formatted_results.append({
                    'jan': result.get('jan', ''),
                    'product_name': result.get('product_name', ''),
                    'standard': result.get('standard', ''),
                    'manufacturer': result.get('manufacturer', ''),
                    'predicted_category': result.get('predicted_category', ''),
                    'predicted_subcategory': result.get('predicted_subcategory', ''),
                    'predicted_segment': result.get('predicted_segment', ''),
                    'predicted_subsegment': result.get('predicted_subsegment', ''),
                    'confidence': result.get('confidence', 0.0),
                    'status': result.get('status', ''),
                    'method': result.get('method', '')
                })
            
            logger.info(f"階層型分類処理完了: 総件数={len(all_results)}, JAN一致={jan_matched_count}, 階層型MLプレディクション={len(ml_results_list)}")
            
            # レスポンス作成（精度検証結果含む）
            response_data = {
                'error': False,
                'message': '階層型分類処理が完了しました',
                'download_id': process_id,
                'statistics': statistics,
                'all_results': formatted_results,
                'result_filename': result_filename,
                'validation_results': validation_results  # 精度検証結果追加
            }
            
            return jsonify(response_data)
            
        finally:
            # アップロード一時ファイル削除
            for temp_file in [market_path, trial_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
    except Exception as e:
        logger.error(f"商品分類処理エラー: {e}")
        return jsonify({'error': True, 'message': f'処理中にエラーが発生しました: {str(e)}'})

def load_file(file_path):
    """ファイルを読み込む（CSV/Excel対応）"""
    try:
        if file_path.endswith('.csv'):
            # CSV文字コード自動判定
            try:
                return pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(file_path, encoding='shift_jis')
        else:
            return pd.read_excel(file_path)
    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {e}")
        raise

def download_classification_results(download_id):
    """分類結果をダウンロード"""
    try:
        # 結果ファイルを検索
        result_files = [f for f in os.listdir(RESULTS_FOLDER) if f.startswith(download_id)]
        
        if not result_files:
            return jsonify({'error': True, 'message': 'ダウンロードファイルが見つかりません'})
        
        result_file = result_files[0]
        result_path = os.path.join(RESULTS_FOLDER, result_file)
        
        if not os.path.exists(result_path):
            return jsonify({'error': True, 'message': 'ダウンロードファイルが見つかりません'})
        
        # ファイル名から処理IDを除去
        clean_filename = result_file.replace(f"{download_id}_", "")
        
        return send_file(
            result_path,
            as_attachment=True,
            download_name=clean_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        logger.error(f"ダウンロードエラー: {e}")
        return jsonify({'error': True, 'message': 'ダウンロード中にエラーが発生しました'})

def cleanup_classification_results(download_id):
    """一時ファイルクリーンアップ"""
    try:
        # 結果ファイル削除
        result_files = [f for f in os.listdir(RESULTS_FOLDER) if f.startswith(download_id)]
        
        deleted_count = 0
        for result_file in result_files:
            result_path = os.path.join(RESULTS_FOLDER, result_file)
            if os.path.exists(result_path):
                os.remove(result_path)
                deleted_count += 1
        
        logger.info(f"クリーンアップ完了: {deleted_count}ファイル削除")
        
        return jsonify({
            'error': False,
            'message': f'{deleted_count}ファイルを削除しました'
        })
        
    except Exception as e:
        logger.error(f"クリーンアップエラー: {e}")
        return jsonify({'error': True, 'message': 'クリーンアップ中にエラーが発生しました'})