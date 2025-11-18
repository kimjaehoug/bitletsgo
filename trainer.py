"""
모델 학습 모듈
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional, Dict, Any
import os
from datetime import datetime


class ModelTrainer:
    """모델 학습 클래스"""
    
    def __init__(self, 
                 model: keras.Model,
                 model_save_path: str = 'models',
                 early_stopping_patience: int = 10,
                 reduce_lr_patience: int = 5):
        """
        Args:
            model: 학습할 모델
            model_save_path: 모델 저장 경로
            early_stopping_patience: Early stopping patience
            reduce_lr_patience: Learning rate reduction patience
        """
        self.model = model
        self.model_save_path = model_save_path
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.history = None
        
        # 모델 저장 디렉토리 생성
        os.makedirs(model_save_path, exist_ok=True)
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = 100,
             batch_size: int = 32,
             verbose: int = 1) -> Dict[str, Any]:
        """
        모델 학습
        
        Args:
            X_train: 학습 입력 데이터
            y_train: 학습 타겟 데이터
            X_val: 검증 입력 데이터
            y_val: 검증 타겟 데이터
            epochs: 에포크 수
            batch_size: 배치 크기
            verbose: 출력 상세도
        
        Returns:
            학습 히스토리
        """
        # 콜백 설정
        callbacks = self._create_callbacks()
        
        # 학습
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False  # 시계열 데이터는 셔플하지 않음
        )
        
        return self.history.history
    
    def _create_callbacks(self) -> list:
        """학습 콜백 생성"""
        callbacks = []
        
        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning Rate Reduction (발산 방지)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
            cooldown=2  # 학습률 감소 후 2 epoch 대기
        )
        callbacks.append(reduce_lr)
        
        # 발산 감지 및 중단 (val_loss가 너무 크면 중단)
        class DivergenceStopping(keras.callbacks.Callback):
            def __init__(self, threshold=10.0):
                super().__init__()
                self.threshold = threshold
                self.best_val_loss = float('inf')
            
            def on_epoch_end(self, epoch, logs=None):
                val_loss = logs.get('val_loss')
                if val_loss is not None:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                    # val_loss가 최고값의 3배 이상이면 발산으로 간주
                    if val_loss > self.best_val_loss * 3.0:
                        print(f"\n경고: 학습이 발산하고 있습니다. val_loss: {val_loss:.4f} (최고: {self.best_val_loss:.4f})")
                        if val_loss > self.threshold:
                            print("학습을 중단합니다.")
                            self.model.stop_training = True
        
        divergence_stopping = DivergenceStopping(threshold=10.0)
        callbacks.append(divergence_stopping)
        
        # Model Checkpoint
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(
            self.model_save_path,
            f'best_model_{timestamp}.h5'
        )
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        return callbacks
    
    def save_model(self, filepath: str):
        """모델 저장"""
        self.model.save(filepath)
        print(f"모델이 저장되었습니다: {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        self.model = keras.models.load_model(filepath)
        print(f"모델이 로드되었습니다: {filepath}")
        return self.model


if __name__ == "__main__":
    # 테스트 코드
    from model import PatchCNNBiLSTM
    
    # 모델 생성
    model_builder = PatchCNNBiLSTM(
        input_shape=(60, 50),
        num_features=50
    )
    model = model_builder.build_model()
    
    # 학습기 생성
    trainer = ModelTrainer(model, model_save_path='models')
    
    # 샘플 데이터 생성
    X_train = np.random.randn(100, 60, 50)
    y_train = np.random.randn(100)
    X_val = np.random.randn(20, 60, 50)
    y_val = np.random.randn(20)
    
    # 학습 (짧게 테스트)
    print("학습 시작...")
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=5,
        batch_size=16,
        verbose=1
    )
    
    print("\n학습 완료!")
    print(f"최종 train loss: {history['loss'][-1]:.4f}")
    print(f"최종 val loss: {history['val_loss'][-1]:.4f}")

