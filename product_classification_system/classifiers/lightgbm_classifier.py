#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBMClassifier - LightGBMを使用した商品分類器
"""

from typing import Tuple, Optional
import numpy as np
import logging

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_classifier import BaseClassifier

logger = logging.getLogger(__name__)


class LightGBMClassifier(BaseClassifier):
    """LightGBMを使用した商品分類器

    特徴：
    - 大量データに強い（高速）
    - カテゴリカル変数を直接扱える
    - 特徴量重要度をネイティブサポート
    - 欠損値を自動処理
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        random_state: int = 42,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        class_weight: Optional[str] = 'balanced',
        verbose: int = -1
    ):
        """初期化

        Args:
            n_estimators: ブースティング回数
            max_depth: 木の最大深さ（-1で無制限）
            learning_rate: 学習率
            random_state: 乱数シード
            num_leaves: 葉の最大数
            min_child_samples: 葉の最小サンプル数
            class_weight: クラス重み
            verbose: ログ出力レベル
        """
        super().__init__()

        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBMがインストールされていません。\n"
                "以下のコマンドでインストールしてください：\n"
                "pip install lightgbm"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.class_weight = class_weight
        self.verbose = verbose

    def train_single_level(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_col: str
    ) -> None:
        """単一階層のLightGBMモデルを訓練

        Args:
            X: 特徴量行列
            y: ラベルエンコードされた目的変数
            target_col: 目的変数のカラム名
        """
        try:
            # クラス重みの計算
            is_unbalance = self.class_weight == 'balanced'

            # LightGBMモデル作成
            model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                is_unbalance=is_unbalance,
                verbose=self.verbose,
                force_col_wise=True  # 警告抑制
            )

            # 訓練
            model.fit(X, y)

            # モデル保存
            self.models[target_col] = model

            logger.debug(f"{target_col}: LightGBMモデル訓練完了（サンプル数={len(X)}, 特徴量数={X.shape[1]}）")

        except Exception as e:
            logger.error(f"LightGBM訓練エラー（{target_col}）: {e}")
            raise

    def predict_single_level(
        self,
        X: np.ndarray,
        target_col: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """単一階層の予測を実行

        Args:
            X: 特徴量行列
            target_col: 予測対象のカラム名

        Returns:
            Tuple[predictions, confidences]: 予測結果と信頼度
        """
        try:
            model = self.models[target_col]

            # 予測
            predictions = model.predict(X)

            # 信頼度計算（確率の最大値）
            probabilities = model.predict_proba(X)
            confidences = np.max(probabilities, axis=1)

            return predictions, confidences

        except Exception as e:
            logger.error(f"LightGBM予測エラー（{target_col}）: {e}")
            raise

    def get_feature_importance(self, target_col: str) -> Optional[np.ndarray]:
        """特徴量重要度を取得

        LightGBMの場合、Gain（情報利得）ベースの重要度を使用

        Args:
            target_col: 対象カラム名

        Returns:
            特徴量重要度の配列
        """
        try:
            model = self.models.get(target_col)
            if model is None:
                return None

            # 特徴量重要度を取得
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_

                # 正規化（合計が1になるように）
                importance_sum = np.sum(importance)
                if importance_sum > 0:
                    importance = importance / importance_sum

                return importance
            else:
                return None

        except Exception as e:
            logger.warning(f"特徴量重要度取得エラー（{target_col}）: {e}")
            return None
