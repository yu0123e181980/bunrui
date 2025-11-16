#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaseClassifier - 商品分類器の抽象基底クラス
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class BaseClassifier(ABC):
    """商品分類器の抽象基底クラス

    すべての分類器はこのクラスを継承し、必要なメソッドを実装する必要があります。
    """

    def __init__(self):
        """初期化"""
        self.models: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_trained: bool = False
        self.feature_importances: Dict[str, np.ndarray] = {}

    @abstractmethod
    def train_single_level(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_col: str
    ) -> None:
        """単一階層のモデルを訓練

        Args:
            X: 特徴量行列
            y: ラベルエンコードされた目的変数
            target_col: 目的変数のカラム名
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_feature_importance(self, target_col: str) -> Optional[np.ndarray]:
        """特徴量重要度を取得

        Args:
            target_col: 対象カラム名

        Returns:
            特徴量重要度の配列（取得できない場合はNone）
        """
        pass

    def train(
        self,
        X: np.ndarray,
        train_df: pd.DataFrame,
        target_cols: List[str]
    ) -> None:
        """全階層のモデルを訓練

        Args:
            X: 特徴量行列
            train_df: 訓練データフレーム
            target_cols: 目的変数のカラムリスト
        """
        try:
            for target_col in target_cols:
                # ラベルエンコーディング
                le = LabelEncoder()
                y = le.fit_transform(train_df[target_col].fillna('不明'))
                self.label_encoders[target_col] = le

                # 単一階層の訓練
                self.train_single_level(X, y, target_col)

                # 特徴量重要度を取得
                importance = self.get_feature_importance(target_col)
                if importance is not None:
                    self.feature_importances[target_col] = importance

                logger.info(f"{target_col}のモデル訓練完了（{self.__class__.__name__}）")

            self.is_trained = True
            logger.info(f"全階層の訓練完了: {len(target_cols)}モデル")

        except Exception as e:
            logger.error(f"モデル訓練エラー: {e}")
            raise

    def predict(
        self,
        X: np.ndarray,
        target_cols: List[str]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """全階層の予測を実行

        Args:
            X: 特徴量行列
            target_cols: 予測対象のカラムリスト

        Returns:
            Tuple[predictions_dict, confidences]: 予測結果の辞書と信頼度配列
        """
        try:
            if not self.is_trained:
                raise ValueError("モデルが訓練されていません")

            predictions_dict = {}
            all_confidences = []

            for target_col in target_cols:
                predictions, confidences = self.predict_single_level(X, target_col)

                # デコード
                decoded_predictions = self.label_encoders[target_col].inverse_transform(predictions)
                predictions_dict[target_col] = decoded_predictions
                all_confidences.append(confidences)

            # 平均信頼度
            avg_confidences = np.mean(all_confidences, axis=0)

            return predictions_dict, avg_confidences

        except Exception as e:
            logger.error(f"予測エラー: {e}")
            raise

    def get_all_feature_importances(self) -> Dict[str, np.ndarray]:
        """全階層の特徴量重要度を取得

        Returns:
            階層ごとの特徴量重要度の辞書
        """
        return self.feature_importances.copy()

    @classmethod
    def get_algorithm_name(cls) -> str:
        """アルゴリズム名を取得

        Returns:
            アルゴリズム名の文字列
        """
        return cls.__name__.replace('Classifier', '')
