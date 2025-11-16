#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LinearSVCClassifier - LinearSVCを使用した商品分類器
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.svm import LinearSVC
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_classifier import BaseClassifier

logger = logging.getLogger(__name__)


class LinearSVCClassifier(BaseClassifier):
    """LinearSVCを使用した商品分類器

    特徴：
    - テキストデータに強い
    - 高次元データに対応
    - クラス不均衡に対応（class_weight='balanced'）
    - 特徴量重要度は係数の絶対値から算出
    """

    def __init__(
        self,
        max_iter: int = 2000,
        random_state: int = 42,
        class_weight: str = 'balanced'
    ):
        """初期化

        Args:
            max_iter: 最大イテレーション数
            random_state: 乱数シード
            class_weight: クラス重み（'balanced'でクラス不均衡に対応）
        """
        super().__init__()
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight

    def train_single_level(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_col: str
    ) -> None:
        """単一階層のLinearSVCモデルを訓練

        Args:
            X: 特徴量行列
            y: ラベルエンコードされた目的変数
            target_col: 目的変数のカラム名
        """
        try:
            # LinearSVCモデル作成
            model = LinearSVC(
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=self.max_iter,
                dual=False  # 特徴量数がサンプル数より多い場合のため
            )

            # 訓練
            model.fit(X, y)

            # モデル保存
            self.models[target_col] = model

            logger.debug(f"{target_col}: LinearSVCモデル訓練完了（サンプル数={len(X)}, 特徴量数={X.shape[1]}）")

        except Exception as e:
            logger.error(f"LinearSVC訓練エラー（{target_col}）: {e}")
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

            # 信頼度計算（決定関数の値をシグモイド変換）
            decision_scores = model.decision_function(X)

            if decision_scores.ndim > 1:
                # 多クラスの場合、最大スコアを正規化して信頼度とする
                max_scores = np.max(decision_scores, axis=1)
                confidences = 1 / (1 + np.exp(-max_scores))  # シグモイド変換
            else:
                # 2クラスの場合
                confidences = 1 / (1 + np.exp(-np.abs(decision_scores)))

            return predictions, confidences

        except Exception as e:
            logger.error(f"LinearSVC予測エラー（{target_col}）: {e}")
            raise

    def get_feature_importance(self, target_col: str) -> Optional[np.ndarray]:
        """特徴量重要度を取得

        LinearSVCの場合、係数の絶対値を重要度とする

        Args:
            target_col: 対象カラム名

        Returns:
            特徴量重要度の配列
        """
        try:
            model = self.models.get(target_col)
            if model is None:
                return None

            # 係数の絶対値を重要度とする
            if hasattr(model, 'coef_'):
                if model.coef_.ndim > 1:
                    # 多クラスの場合、各クラスの係数の平均絶対値を使用
                    importance = np.mean(np.abs(model.coef_), axis=0)
                else:
                    importance = np.abs(model.coef_)

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
