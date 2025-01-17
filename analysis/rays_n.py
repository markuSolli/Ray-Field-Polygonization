import csv

DIR_PATH = 'analysis/data/rays_n'

def compute_values() -> tuple[list[int], list[int]]:
    n_values = list(range(50, 501, 50))
    rays_n = []

    for n in n_values:
        ray_n = n * (n - 1)
        rays_n.append(ray_n)
    
    return n_values, rays_n

def save_results(n_values: list[int], rays_n: list[int]) -> None:
    with open(f'{DIR_PATH}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(n_values)
        writer.writerow(rays_n)

n_values, rays_n = compute_values()
save_results(n_values, rays_n)