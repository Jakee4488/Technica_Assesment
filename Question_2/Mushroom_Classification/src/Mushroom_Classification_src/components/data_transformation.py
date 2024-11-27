import os
import pandas as pd
from Mushroom_Classification_src import logger
from Mushroom_Classification_src.config.configuration import ConfigurationManager
from Mushroom_Classification_src.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


class DataTransformation:
    def __init__(self, config: DataTransformationConfig,test_size=0.3, random_state=42, pca_components=2):
        self.config = config
        self.test_size = test_size
        self.random_state = random_state
        self.pca_components = pca_components


    def load_data(self):
        """Load data from the specified path."""
        data = pd.read_csv(self.config.data_path)
        logger.info("Loaded data with shape: %s", data.shape)
        return data

    def encode_features(self, features, target):
        """Apply Label Encoding to features and target."""
        encoder_features = LabelEncoder()
        for col in features.columns:
            features[col] = encoder_features.fit_transform(features[col])
        
        encoder_target = LabelEncoder()
        target = encoder_target.fit_transform(target)
        logger.info("Features and target encoded.")
        return features, target

    def scale_features(self, features_train, features_test):
        """Scale features using StandardScaler."""
        scaler = StandardScaler()
        scaler.fit(features_train)
        features_train_transformed = scaler.transform(features_train)
        features_test_transformed = scaler.transform(features_test)
        logger.info("Features scaled.")
        return features_train_transformed, features_test_transformed

    def apply_pca(self, features_train_transformed, features_test_transformed):
        """Apply PCA transformation to the data."""
        pca_transformer = PCA(n_components=self.pca_components)
        train_pca = pca_transformer.fit_transform(features_train_transformed)
        test_pca = pca_transformer.transform(features_test_transformed)
        logger.info("PCA transformation applied. Original shape: %s, PCA shape: %s",
                    features_train_transformed.shape, train_pca.shape)
        return train_pca, test_pca

    def visualize_pca(self, train_pca):
        """Visualize PCA-transformed data."""
        plt.scatter(train_pca[:, 0], train_pca[:, 1], alpha=0.2)
        plt.title("PCA-transformed Data")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()

    def save_transformed_data(self, train_pca, test_pca, target_train, target_test):
        """Save PCA-transformed data and targets to CSV files."""
        np.savetxt(os.path.join(self.config.root_dir, "train_pca.csv"), train_pca, delimiter=",")
        np.savetxt(os.path.join(self.config.root_dir, "test_pca.csv"), test_pca, delimiter=",")
        np.savetxt(os.path.join(self.config.root_dir, "target_train.csv"), target_train, delimiter=",")
        np.savetxt(os.path.join(self.config.root_dir, "target_test.csv"), target_test, delimiter=",")
        logger.info("Transformed data saved as CSV files.")

    def transform(self):
        """Full data transformation pipeline."""
        data = self.load_data()
        
        # Assuming 'class' column is the label
        features = data.drop(columns=["class"])
        target = data["class"]

        # Encode features and target
        features, target = self.encode_features(features, target)

        # One-hot encode features
        features = pd.get_dummies(features, columns=features.columns, drop_first=True)

        # Train-test split
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=self.test_size, random_state=self.random_state
        )

        # Scale features
        features_train_transformed, features_test_transformed = self.scale_features(features_train, features_test)

        # PCA Transformation
        train_pca, test_pca = self.apply_pca(features_train_transformed, features_test_transformed)

        # Visualize PCA-transformed data
        self.visualize_pca(train_pca)

        # Save the transformed data
        self.save_transformed_data(train_pca, test_pca, target_train, target_test)

        return train_pca, test_pca, target_train, target_test
