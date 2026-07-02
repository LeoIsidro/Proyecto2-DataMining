import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, variance_threshold=0.90):
        """
        Principal Component Analysis implementado desde cero usando numpy.
        :param variance_threshold: Fracción de varianza a retener (ej. 0.90 para 90%)
        """
        self.variance_threshold = variance_threshold
        self.components = None
        self.eigenvalues = None
        self.mean = None
        self.std = None
        self.explained_variance_ratio = None
        self.k_components = None

    def fit(self, X):
        # 1. Estandarizar
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Prevenir división por cero si alguna característica es constante
        self.std[self.std == 0] = 1.0
        X_scaled = (X - self.mean) / self.std

        # 2. Matriz de covarianza
        # X_scaled shape es (n_samples, n_features)
        n_samples = X.shape[0]
        cov_matrix = (X_scaled.T @ X_scaled) / (n_samples - 1)

        # 3. Eigenvalores y eigenvectores
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Ordenar de mayor a menor
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.components = eigenvectors[:, sorted_indices]

        # 5. Calcular varianza explicada y k para threshold
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues / total_variance
        cumulative_variance = np.cumsum(self.explained_variance_ratio)
        
        # Encontrar el menor k que supera el threshold
        self.k_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1

    def transform(self, X, k=None):
        if k is None:
            k = self.k_components
        X_scaled = (X - self.mean) / self.std
        # Proyección a k dimensiones
        return X_scaled @ self.components[:, :k]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def plot_variance(self):
        cumulative_variance = np.cumsum(self.explained_variance_ratio)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
        plt.axhline(y=self.variance_threshold, color='r', linestyle='-', label=f'Threshold ({self.variance_threshold*100}%)')
        plt.axvline(x=self.k_components, color='r', linestyle='--', label=f'k={self.k_components}')
        plt.title('Varianza Acumulada Explicada por Componentes Principales')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Varianza Acumulada Explicada')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_2d(self, X_transformed, labels=None, title='PCA - Proyección 2D'):
        plt.figure(figsize=(10, 8))
        if labels is not None:
            scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter)
        else:
            plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6)
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.title(title)
        plt.grid(True)
        plt.show()

    def interpret_components(self, feature_names, n_top_features=5):
        """
        Interpreta qué variables originales contribuyen más a los componentes principales.
        """
        print(f"Interpretación de los primeros {self.k_components} componentes principales:")
        for i in range(self.k_components):
            component = self.components[:, i]
            # Ordenar por valor absoluto de contribución
            top_indices = np.argsort(np.abs(component))[::-1][:n_top_features]
            print(f"CP {i+1}:")
            for idx in top_indices:
                weight = component[idx]
                print(f"  - {feature_names[idx]}: {weight:.4f}")
            print()


class SVD:
    def __init__(self):
        """
        Singular Value Decomposition para reducción de dimensionalidad de matrices dispersas (como TF-IDF o User-Item).
        """
        self.U = None
        self.S = None
        self.Vt = None
        
    def fit(self, X):
        # Descomposición exacta usando algebra lineal base de numpy
        self.U, self.S, self.Vt = np.linalg.svd(X, full_matrices=False)
        
    def transform(self, k):
        """
        Devuelve la matriz proyectada en k dimensiones (embeddings).
        """
        return self.U[:, :k] @ np.diag(self.S[:k])
        
    def reconstruct(self, k):
        """
        Reconstruye la matriz original a partir de k componentes.
        """
        return self.U[:, :k] @ np.diag(self.S[:k]) @ self.Vt[:k, :]
        
    def evaluate_reconstruction_error(self, X, max_k=None, step=1):
        if max_k is None:
            max_k = min(X.shape)
            
        errors = []
        ks = list(range(1, max_k + 1, step))
        norm_X = np.linalg.norm(X, 'fro')
        
        for k in ks:
            X_approx = self.reconstruct(k)
            # Norma de Frobenius del error (error relativo)
            error = np.linalg.norm(X - X_approx, 'fro') / norm_X
            errors.append(error)
            
        plt.figure(figsize=(10, 6))
        plt.plot(ks, errors, marker='o')
        plt.title('Error de Reconstrucción vs. k Dimensiones Retenidas (SVD)')
        plt.xlabel('k (Número de Dimensiones)')
        plt.ylabel('Error Relativo (Norma de Frobenius)')
        plt.grid(True)
        plt.show()
        
        return ks, errors
