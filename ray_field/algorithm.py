from abc import ABC, abstractmethod
from ray_field import CheckpointName
from open3d.geometry import TriangleMesh

class Algorithm(ABC):
    poisson_depth: int = 8
    chamfer_samples: int = 30000
    time_samples: int = 5

    @staticmethod
    @abstractmethod
    def surface_reconstruction(model_name: CheckpointName, N: int) -> TriangleMesh:
        """Perform a surface reconstruction using the class' algorithm.

        Args:
            model_name (CheckpointName): Valid model name
            N (int): Number of origin points, resulting number of rays is N * (N - 1)

        Returns:
            TriangleMesh: The reconstructred mesh
        """        
        pass

    @staticmethod
    @abstractmethod
    def hit_rate(model_name: CheckpointName, N_values: list[int]) -> list[float]:
        """Calculate the number of rays that hit the model divided by the total number of rays that are generated.

        Args:
            model_name (CheckpointName): Valid model name
            N_values (list[int]): N-values to test

        Returns:
            list[float]: Hit rates corresponding to the list of N-values.
            From 0 to 1.
        """        
        pass

    @staticmethod
    @abstractmethod
    def chamfer(model_name: CheckpointName, N_values: list[int]) -> list[float]:
        """Calculate the Chamfer Distance between the reconstructed surface and the Stanford mesh.
        Will sample both surfaces with M points and compare those.

        Args:
            model_name (CheckpointName): Valid model name
            N_values (list[int]): N-values to test

        Returns:
            list[float]: Distances corresponding to the list of N-values.
        """        
        pass

    @staticmethod
    @abstractmethod
    def hausdorff(model_name: CheckpointName, N_values: list[int]) -> list[float]:
        """Calculate the Hausdorff Distance between the reconstructed surface and the Stanford mesh.
        Will compare the vertices of the meshes.

        Args:
            model_name (CheckpointName): Valid model name
            N_values (list[int]): N-values to test

        Returns:
            list[float]: Distances corresponding to the list of N-values.
        """        
        pass

    @staticmethod
    @abstractmethod
    def time(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[int]]:
        """Measure exectution time of the surface reconstruction.

        Args:
            model_name (CheckpointName): Valid model name
            N_values (list[int]): N-values to test 

        Returns:
            tuple[list[float], list[int]]
                - Execution times
                - Number of rays 
        """        
        pass

    @staticmethod
    @abstractmethod
    def time_steps(model_name: CheckpointName, N_values: list[int]) -> list[list[float]]:
        """Measure execution time of the surface reconstruction divided in several steps

        Args:
            model_name (CheckpointName): Valid model name
            N_values (list[int]): N-values to test

        Returns:
            list[list[float]]: (N, M) execution times, where M is the number of steps
        """        
        pass

    @staticmethod
    @abstractmethod
    def time_chamfer(model_name: CheckpointName, N_values: list[int]) -> tuple[list[float], list[float]]:
        """Measures the chamfer distance and the execution time.

        Args:
            model_name (CheckpointName): Valid model name
            N_values (list[int]): N-values to test

        Returns:
            tuple[list[float], list[float]]
                - Execution times
                - Chamfer distances
        """        
        pass
