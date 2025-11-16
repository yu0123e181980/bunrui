#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品分類システム - Webアプリケーション版 V2
改良版: アルゴリズム自動選択 + マルチプロセス並列処理 + 特徴量重要度可視化 + SHAP分析
"""

from flask import request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
import uuid
import logging
from datetime import datetime
import warnings
import tempfile
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import gc
from multiprocessing import Pool, cpu_count
import functools

# 自作モジュールのインポート
from base_classifier import BaseClassifier
from classifiers import LinearSVCClassifier, LightGBMClassifier

warnings.filterwarnings('ignore')

# ログ設定
logger = logging.getLogger(__name__)

# 設定
UPLOAD_FOLDER = 'temp_uploads'
RESULTS_FOLDER = 'temp_results'
ALGORITHM_THRESHOLD = 500  # アルゴリズム自動選択の閾値（データ件数）
PARALLEL_BATCH_SIZE = 100  # 並列処理のバッチサイズ

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
    """Webアプリケーション用商品分類エンジン（階層型学習対応・改良版）

    新機能:
    - アルゴリズム自動選択（LinearSVC/LightGBM）
    - マルチプロセス並列処理
    - 特徴量重要度可視化
    - SHAP値分析（骨格のみ）
    - メモリ効率化
    """

    def __init__(self, algorithm: Optional[str] = None):
        """初期化

        Args:
            algorithm: 使用するアルゴリズム（'linear_svc', 'lightgbm', None=自動選択）
        """
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[BaseClassifier] = None
        self.jan_dict: Dict[str, Dict[str, str]] = {}
        self.is_trained: bool = False
        self.algorithm: Optional[str] = algorithm

        # 階層型学習用のマッピング
        self.hierarchy_mapping: Dict[str, Dict[str, List[str]]] = {}
        self.use_hierarchical: bool = True

        # 特徴量名保存（可視化用）
        self.feature_names: List[str] = []

    def normalize_text(self, text: str) -> str:
        """日本語テキストの正規化処理"""
        if pd.isna(text) or text == '':
            return ''

        text = str(text)
        text = text.lower()
        return text.strip()

    def extract_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """数値特徴量を抽出（容量・入数・価格）"""
        result_df = df.copy()

        # 価格の数値化
        if 'avg_price' in df.columns:
            def extract_price(price_str):
                if pd.isna(price_str) or price_str == '':
                    return 0
                matches = re.findall(r'[\d,]+', str(price_str))
                if matches:
                    try:
                        return float(matches[0].replace(',', ''))
                    except:
                        return 0
                return 0

            result_df['price_numeric'] = df['avg_price'].apply(extract_price)

        # 容量の数値化
        if 'standard' in df.columns:
            def extract_volume(standard_str):
                if pd.isna(standard_str) or standard_str == '':
                    return 0
                matches = re.findall(r'(\d+(?:\.\d+)?)\s*(ml|g|kg|l)', str(standard_str).lower())
                if matches:
                    value, unit = matches[0]
                    value = float(value)
                    if unit in ['kg']:
                        value *= 1000
                    elif unit in ['l']:
                        value *= 1000
                    return value
                return 0

            def extract_count(standard_str):
                if pd.isna(standard_str) or standard_str == '':
                    return 1
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

    def create_hierarchy_mapping(self, trial_df: pd.DataFrame) -> None:
        """階層構造マッピングを作成"""
        try:
            trial_df.columns = trial_df.columns.str.strip()

            logger.info(f"利用可能なカラム名: {list(trial_df.columns)}")

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

    def get_valid_choices_for_hierarchy(self, parent_prediction: str, hierarchy_level: str) -> List[str]:
        """指定された上位階層に基づいて有効な選択肢を取得"""
        mapping_key = {
            'subcategory': 'category_to_subcategory',
            'segment': 'subcategory_to_segment',
            'subsegment': 'segment_to_subsegment'
        }.get(hierarchy_level)

        if mapping_key and parent_prediction in self.hierarchy_mapping[mapping_key]:
            return self.hierarchy_mapping[mapping_key][parent_prediction]
        return []

    def find_closest_valid_choice(
        self,
        prediction: str,
        valid_choices: List[str],
        label_encoder: LabelEncoder
    ) -> Tuple[str, float]:
        """制約に合わない予測結果を最も近い有効な選択肢に修正"""
        if not valid_choices or prediction in valid_choices:
            return prediction, 1.0

        try:
            best_choice = valid_choices[0]
            best_score = 0

            for choice in valid_choices:
                common_chars = set(prediction) & set(choice)
                similarity = len(common_chars) / max(len(prediction), len(choice), 1)

                if similarity > best_score:
                    best_score = similarity
                    best_choice = choice

            confidence_penalty = max(0.3, best_score)

            return best_choice, confidence_penalty

        except Exception as e:
            logger.warning(f"最適選択肢検索エラー: {e}")
            return valid_choices[0] if valid_choices else prediction, 0.3

    def prepare_features(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """特徴量を作成（文字n-gram + 数値特徴量）"""
        try:
            # テキスト特徴量の正規化と結合
            text_features = []
            for _, row in df.iterrows():
                text_parts = []
                for col in feature_cols:
                    if col in ['price_numeric', 'volume_numeric', 'count_numeric']:
                        continue
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

                # 特徴量名を保存（可視化用）
                self.feature_names = list(self.vectorizer.get_feature_names_out())
            else:
                text_matrix = self.vectorizer.transform(text_features).toarray()

            # 数値特徴量を追加
            numeric_features = []
            numeric_names = []
            for col in ['price_numeric', 'volume_numeric', 'count_numeric']:
                if col in df.columns:
                    values = df[col].fillna(0).values
                    if np.max(values) > 0:
                        values = values / np.max(values)
                    numeric_features.append(values.reshape(-1, 1))
                    numeric_names.append(col)

            if numeric_features:
                numeric_matrix = np.hstack(numeric_features)
                features = np.hstack([text_matrix, numeric_matrix])

                # 数値特徴量名も追加
                if not numeric_names or len(self.feature_names) == len(self.vectorizer.get_feature_names_out()):
                    self.feature_names.extend(numeric_names)
            else:
                features = text_matrix

            return features

        except Exception as e:
            logger.error(f"特徴量作成エラー: {e}")
            raise

    def create_jan_dict(self, trial_df: pd.DataFrame) -> None:
        """JAN辞書を作成"""
        try:
            trial_df.columns = trial_df.columns.str.strip()

            self.jan_dict = {}
            for _, row in trial_df.iterrows():
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

    def select_algorithm(self, train_size: int) -> BaseClassifier:
        """データサイズに応じてアルゴリズムを自動選択

        Args:
            train_size: 訓練データのサンプル数

        Returns:
            選択されたクラシファイア
        """
        if self.algorithm == 'linear_svc':
            logger.info("アルゴリズム選択: LinearSVC（手動指定）")
            return LinearSVCClassifier()
        elif self.algorithm == 'lightgbm':
            logger.info("アルゴリズム選択: LightGBM（手動指定）")
            return LightGBMClassifier()
        else:
            # 自動選択：500件未満はLinearSVC、500件以上はLightGBM
            if train_size < ALGORITHM_THRESHOLD:
                logger.info(f"アルゴリズム自動選択: LinearSVC（サンプル数={train_size} < {ALGORITHM_THRESHOLD}）")
                return LinearSVCClassifier()
            else:
                logger.info(f"アルゴリズム自動選択: LightGBM（サンプル数={train_size} >= {ALGORITHM_THRESHOLD}）")
                try:
                    return LightGBMClassifier()
                except ImportError:
                    logger.warning("LightGBMが利用不可のため、LinearSVCにフォールバック")
                    return LinearSVCClassifier()

    def train(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        use_hierarchical: bool = True
    ) -> None:
        """モデルを訓練

        Args:
            train_df: 訓練データフレーム
            feature_cols: 特徴量カラムのリスト
            target_cols: 目的変数カラムのリスト
            use_hierarchical: 階層型学習を使用するか
        """
        try:
            # 有効なデータのみ使用
            train_df = train_df.dropna(subset=target_cols)
            if len(train_df) == 0:
                raise ValueError("訓練データが空です")

            # アルゴリズム選択
            self.classifier = self.select_algorithm(len(train_df))

            # 階層型学習の設定
            if use_hierarchical and not self.hierarchy_mapping:
                logger.warning("階層マッピングが未作成のため従来型学習に切り替えます")
                use_hierarchical = False

            self.use_hierarchical = use_hierarchical

            # 数値特徴量を抽出
            train_df = self.extract_numeric_features(train_df)

            # 特徴量作成
            X = self.prepare_features(train_df, feature_cols)

            # モデル訓練
            self.classifier.train(X, train_df, target_cols)

            self.is_trained = True

            hierarchical_info = "階層型" if use_hierarchical else "従来型"
            algorithm_name = self.classifier.get_algorithm_name()
            logger.info(f"モデル訓練完了（{hierarchical_info}, {algorithm_name}）: {len(train_df)}件")

            # メモリ解放
            gc.collect()

        except Exception as e:
            logger.error(f"モデル訓練エラー: {e}")
            raise

    def predict_batch(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        use_hierarchical: bool = True,
        batch_size: int = 100,
        use_parallel: bool = True
    ) -> pd.DataFrame:
        """バッチ予測を実行

        Args:
            df: 予測対象データフレーム
            feature_cols: 特徴量カラムのリスト
            target_cols: 目的変数カラムのリスト
            use_hierarchical: 階層型予測を使用するか
            batch_size: バッチサイズ
            use_parallel: 並列処理を使用するか

        Returns:
            予測結果を含むデータフレーム
        """
        try:
            if not self.is_trained:
                raise ValueError("モデルが訓練されていません")

            use_hierarchical = use_hierarchical and bool(self.hierarchy_mapping)

            # 数値特徴量を抽出
            df = self.extract_numeric_features(df)

            total_rows = len(df)

            if use_parallel and total_rows > batch_size * 2:
                # 並列処理を使用
                logger.info(f"マルチプロセス並列予測開始（バッチサイズ={batch_size}）")
                results = self._predict_parallel(df, feature_cols, target_cols, use_hierarchical, batch_size)
            else:
                # シングルプロセスで予測
                results = []
                for i in range(0, total_rows, batch_size):
                    batch = df.iloc[i:i+batch_size]
                    X = self.prepare_features(batch, feature_cols)

                    batch_results = batch.copy()

                    if use_hierarchical:
                        batch_results = self._hierarchical_predict(batch_results, X, target_cols)
                    else:
                        batch_results = self._standard_predict(batch_results, X, target_cols)

                    results.append(batch_results)

                    # メモリ解放
                    del X
                    gc.collect()

            prediction_type = "階層型" if use_hierarchical else "従来型"
            logger.info(f"{prediction_type}予測完了: {total_rows}件")

            result_df = pd.concat(results, ignore_index=True)

            # メモリ解放
            gc.collect()

            return result_df

        except Exception as e:
            logger.error(f"予測エラー: {e}")
            raise

    def _predict_parallel(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        use_hierarchical: bool,
        batch_size: int
    ) -> List[pd.DataFrame]:
        """並列予測を実行（multiprocessing.Pool使用）

        注意: このメソッドは現在シングルプロセスで実行されます。
        マルチプロセス化にはpickle化の問題を解決する必要があります。
        """
        # TODO: マルチプロセス化の実装（現在はシングルプロセスで実行）
        # pickleエラーを回避するため、シングルプロセスで実行
        logger.warning("マルチプロセス予測は現在未実装のため、シングルプロセスで実行します")

        results = []
        total_rows = len(df)

        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i+batch_size]
            X = self.prepare_features(batch, feature_cols)

            batch_results = batch.copy()

            if use_hierarchical:
                batch_results = self._hierarchical_predict(batch_results, X, target_cols)
            else:
                batch_results = self._standard_predict(batch_results, X, target_cols)

            results.append(batch_results)

            # メモリ解放
            del X
            gc.collect()

        return results

    def _standard_predict(
        self,
        batch_results: pd.DataFrame,
        X: np.ndarray,
        target_cols: List[str]
    ) -> pd.DataFrame:
        """従来型予測（階層制約なし）"""
        predictions_dict, confidences = self.classifier.predict(X, target_cols)

        for target_col, predictions in predictions_dict.items():
            batch_results[target_col] = predictions

        batch_results['confidence'] = confidences

        return batch_results

    def _hierarchical_predict(
        self,
        batch_results: pd.DataFrame,
        X: np.ndarray,
        target_cols: List[str]
    ) -> pd.DataFrame:
        """階層型予測（階層制約あり）"""
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
                # 単一サンプル予測
                X_single = X[idx:idx+1]
                predictions_dict, confidences = self.classifier.predict(X_single, [target_col])

                decoded_prediction = predictions_dict[target_col][0]
                base_confidence = confidences[0]

                # 階層制約チェック
                if hierarchy_level:
                    parent_col = hierarchy_order[hierarchy_order.index((target_col, hierarchy_level)) - 1][0]
                    parent_prediction = row_predictions[parent_col]

                    valid_choices = self.get_valid_choices_for_hierarchy(
                        parent_prediction, hierarchy_level
                    )

                    if valid_choices and decoded_prediction not in valid_choices:
                        corrected_prediction, penalty = self.find_closest_valid_choice(
                            decoded_prediction, valid_choices, self.classifier.label_encoders[target_col]
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

    def validate_accuracy(
        self,
        matched_data: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str]
    ) -> Optional[Dict[str, Any]]:
        """精度検証を実行（9段階：10%-90%）"""
        try:
            train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            validation_results = {
                'train_ratios': train_ratios,
                'accuracy_scores': {col: [] for col in target_cols},
                'sample_counts': []
            }

            logger.info("精度検証開始...")

            for i, train_ratio in enumerate(train_ratios):
                logger.info(f"検証 {i+1}/9: 学習データ{int(train_ratio*100)}%")

                # データ分割
                n_samples = len(matched_data)
                n_train = int(n_samples * train_ratio)

                shuffled_data = matched_data.sample(frac=1, random_state=42+i).reset_index(drop=True)
                train_data = shuffled_data.iloc[:n_train]
                test_data = shuffled_data.iloc[n_train:]

                # サブセグメント（最下層）でクラス数をチェック
                subsegment_col = 'predicted_subsegment'
                if subsegment_col in train_data.columns:
                    unique_subsegments = train_data[subsegment_col].nunique()
                    if unique_subsegments < 2:
                        logger.warning(f"検証 {i+1}/9 スキップ: サブセグメントが{unique_subsegments}クラスのみ")
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
                temp_classifier = ProductClassifierWeb(algorithm=self.algorithm)
                temp_classifier.hierarchy_mapping = self.hierarchy_mapping.copy()
                temp_classifier.use_hierarchical = self.use_hierarchical

                # 学習実行
                try:
                    temp_classifier.train(train_data, feature_cols, target_cols, use_hierarchical=self.use_hierarchical)
                except Exception as e:
                    logger.error(f"検証 {i+1}/9 訓練エラー: {str(e)}")
                    for target_col in target_cols:
                        validation_results['accuracy_scores'][target_col].append(0.0)
                    continue

                # 予測実行
                try:
                    predictions = temp_classifier.predict_batch(
                        test_data, feature_cols, target_cols,
                        use_hierarchical=self.use_hierarchical, batch_size=50, use_parallel=False
                    )
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

                # メモリ解放
                del temp_classifier
                gc.collect()

                logger.info(f"検証 {i+1}/9 完了")

            logger.info("精度検証完了")

            # メモリ解放
            gc.collect()

            return validation_results

        except Exception as e:
            logger.error(f"精度検証エラー: {e}")
            return None

    def get_feature_importances(self) -> Optional[Dict[str, Dict[str, float]]]:
        """特徴量重要度を取得（4階層別、上位20件）

        Returns:
            {
                'predicted_category': {'feature_name': importance, ...},
                'predicted_subcategory': {...},
                ...
            }
        """
        try:
            if not self.is_trained or not self.classifier:
                return None

            all_importances = self.classifier.get_all_feature_importances()

            if not all_importances:
                return None

            # 各階層の上位20件を抽出
            result = {}
            for target_col, importances in all_importances.items():
                if importances is None or len(importances) == 0:
                    continue

                # 特徴量名とインデックスをマッピング
                feature_importance_pairs = []
                for i, importance in enumerate(importances):
                    if i < len(self.feature_names):
                        feature_importance_pairs.append((self.feature_names[i], float(importance)))

                # 重要度でソート（降順）
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

                # 上位20件を辞書化
                result[target_col] = dict(feature_importance_pairs[:20])

            return result

        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {e}")
            return None

    def calculate_shap_values(
        self,
        sample_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        max_samples: int = 100
    ) -> Optional[Dict[str, Any]]:
        """SHAP値を計算（骨格実装）

        注意: 完全な実装にはshapライブラリが必要です。
        現在は骨格のみの実装です。

        Args:
            sample_df: サンプルデータフレーム
            feature_cols: 特徴量カラムのリスト
            target_col: 対象カラム
            max_samples: 最大サンプル数

        Returns:
            SHAP値の辞書（未実装の場合はNone）
        """
        try:
            # TODO: shapライブラリのインストールと実装
            logger.warning("SHAP値分析は骨格実装のため、現在は利用できません")
            logger.info("完全な実装には 'pip install shap' が必要です")

            # 骨格実装：特徴量重要度を代替として返す
            importances = self.get_feature_importances()
            if importances and target_col in importances:
                return {
                    'type': 'feature_importance',
                    'target_col': target_col,
                    'importances': importances[target_col],
                    'note': 'SHAP値の代わりに特徴量重要度を表示しています'
                }

            return None

        except Exception as e:
            logger.error(f"SHAP値計算エラー: {e}")
            return None


# ============================================================================
# API関数群
# ============================================================================

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
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"カラム情報取得エラー: {e}")
        return jsonify({'error': True, 'message': str(e)})


def process_product_classification():
    """商品分類処理のメイン関数（CSV出力対応）"""
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

        logger.info(f"分類処理開始（改良版）: {process_id}")

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
            market_df = market_df.rename(columns={jan_column: 'jan', product_column: 'product_name'})

            for old_col, new_col in [
                (standard_column, 'standard'),
                (manufacturer_column, 'manufacturer'),
                (container_column, 'container'),
                (avg_price_column, 'avg_price'),
                (category_column, 'existing_category'),
                (subcategory_column, 'existing_subcategory'),
                (segment_column, 'existing_segment'),
                (subsegment_column, 'existing_subsegment')
            ]:
                if old_col:
                    market_df[new_col] = market_df[old_col]
                else:
                    market_df[new_col] = ''

            # JANを正規化
            market_df['jan'] = market_df['jan'].apply(normalize_jan)
            trial_df['JAN'] = trial_df['JAN'].apply(normalize_jan)

            # 分類器初期化（アルゴリズム自動選択）
            classifier = ProductClassifierWeb()

            # JAN辞書作成
            classifier.create_jan_dict(trial_df)

            # 階層マッピング作成
            if len(trial_df) > 0:
                classifier.create_hierarchy_mapping(trial_df)

            # JAN照合
            jan_matched_list = []
            jan_unmatched_list = []

            for _, row in market_df.iterrows():
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

            # メモリ解放
            gc.collect()

            # 精度検証実行
            validation_results = None
            if len(jan_matched_list) > 50:
                logger.info("精度検証開始...")
                matched_df = pd.DataFrame(jan_matched_list)

                # 利用可能な特徴量を動的に決定
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

            # メモリ解放
            gc.collect()

            # 機械学習による予測
            ml_results_list = []
            feature_importances = None

            if len(jan_unmatched_list) > 0 and len(jan_matched_list) > 0:
                logger.info("機械学習モデル訓練開始...")

                # 訓練データ準備
                train_df = pd.DataFrame(jan_matched_list)

                # 利用可能な特徴量を決定
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

                # モデル訓練
                classifier.train(train_df, available_features, target_cols, use_hierarchical=True)

                # 特徴量重要度を取得
                feature_importances = classifier.get_feature_importances()

                # 予測実行
                logger.info(f"ML予測実行中: {len(jan_unmatched_list)}件...")

                unmatched_df = pd.DataFrame(jan_unmatched_list)

                # 予測用データにも同じ特徴量カラムを確保
                for col in available_features:
                    if col not in unmatched_df.columns:
                        unmatched_df[col] = ''

                predictions_df = classifier.predict_batch(
                    unmatched_df, available_features, target_cols,
                    use_hierarchical=True, batch_size=100, use_parallel=True
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
                    algorithm_name = classifier.classifier.get_algorithm_name()
                    result_row.update({
                        'status': status,
                        'method': f'{algorithm_name}予測'
                    })
                    ml_results_list.append(result_row)

                logger.info(f"ML予測完了: {len(ml_results_list)}件")

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

            # メモリ解放
            gc.collect()

            # 結果統合
            all_results = jan_matched_list + ml_results_list

            if not all_results:
                return jsonify({'error': True, 'message': '処理結果が空です'})

            results_df = pd.DataFrame(all_results)

            # 結果保存（CSV形式）
            result_filename = f"商品分類結果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            result_path = os.path.join(RESULTS_FOLDER, f"{process_id}_{result_filename}")

            # CSV出力用にカラム順序を整理
            output_columns = [
                'jan', 'product_name', 'standard', 'manufacturer',
                'predicted_category', 'predicted_subcategory', 'predicted_segment', 'predicted_subsegment',
                'confidence', 'status', 'method'
            ]

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

            # CSV保存（UTF-8 BOM付き：Excelで文字化けしないように）
            output_df.to_csv(result_path, index=False, encoding='utf-8-sig')

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

            logger.info(f"分類処理完了: 総件数={len(all_results)}, JAN一致={jan_matched_count}, ML予測={len(ml_results_list)}")

            # メモリ解放
            gc.collect()

            # レスポンス作成
            response_data = {
                'error': False,
                'message': '分類処理が完了しました',
                'download_id': process_id,
                'statistics': statistics,
                'all_results': formatted_results,
                'result_filename': result_filename,
                'validation_results': validation_results,
                'feature_importances': feature_importances  # 特徴量重要度追加
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


def load_file(file_path: str) -> pd.DataFrame:
    """ファイルを読み込む（CSV/Excel対応）"""
    try:
        if file_path.endswith('.csv'):
            try:
                return pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(file_path, encoding='shift_jis')
        else:
            return pd.read_excel(file_path)
    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {e}")
        raise


def download_classification_results(download_id: str):
    """分類結果をダウンロード"""
    try:
        result_files = [f for f in os.listdir(RESULTS_FOLDER) if f.startswith(download_id)]

        if not result_files:
            return jsonify({'error': True, 'message': 'ダウンロードファイルが見つかりません'})

        result_file = result_files[0]
        result_path = os.path.join(RESULTS_FOLDER, result_file)

        if not os.path.exists(result_path):
            return jsonify({'error': True, 'message': 'ダウンロードファイルが見つかりません'})

        clean_filename = result_file.replace(f"{download_id}_", "")

        # CSV形式でダウンロード
        return send_file(
            result_path,
            as_attachment=True,
            download_name=clean_filename,
            mimetype='text/csv'
        )

    except Exception as e:
        logger.error(f"ダウンロードエラー: {e}")
        return jsonify({'error': True, 'message': 'ダウンロード中にエラーが発生しました'})


def cleanup_classification_results(download_id: str):
    """一時ファイルクリーンアップ"""
    try:
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
