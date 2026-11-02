from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import random
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

def extract_embeddings(
        data,
        batch_size=1000,
        model_name="sentence-t5-base"):

    model = SentenceTransformer(model_name) 
    model.eval()

    with torch.no_grad():
        sentence_embeddings = []
        for i in tqdm(range(len(data) // batch_size+1)):
            embeddings = model.encode(
                data[i * batch_size:(i + 1) * batch_size])
            if len(embeddings) > 0:
                sentence_embeddings.append(embeddings)
    sentence_embeddings = np.concatenate(sentence_embeddings)
    del model
    return sentence_embeddings

def load_jsonl_texts(jsonl_path, text_key="text"):
    """
    Load texts from a JSONL file.
    """
    texts = []
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj[text_key])
    return texts

def compute_embeddings(datasetname, jsonl_path, output_dir, model_name="sentence-t5-base", batch_size=1000):
    """
    Compute sentence embeddings from a JSONL file and save as .npz.
    Args:
        datasetname: dataset name, used in the output file name
        jsonl_path: path to the input JSONL file
        output_dir: directory to save the .npz file
        model_name: sentence transformer model name
        batch_size: batch size for encoding
    """
    os.makedirs(output_dir, exist_ok=True)
    # Load texts
    texts = load_jsonl_texts(jsonl_path, text_key="text")
    labels = []
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            labels.append(obj["label"][0])
    # Compute embeddings
    embeddings = extract_embeddings(texts, batch_size=batch_size, model_name=model_name)
    # Save labels alongside the embeddings
    additional_info = np.array(labels)
    # Save as .npz
    out_path = os.path.join(output_dir, f"cls_{datasetname}_train_all.embeddings.npz")
    np.savez_compressed(out_path, embeddings=embeddings, additional_info=additional_info)
    print(f"Saved embeddings to {out_path}")




def find_outliers(embeddings,
                  distance_percent=99,
                  lof_n_neighbors=10,):
    print("Finding outliers in embeddings...")
    print("distance_percent:", distance_percent)
    print("lof_n_neighbors:", lof_n_neighbors)
    # Create index array
    indices = np.arange(len(embeddings))
    
    # === Distance to mean (farthest outliers) ===
    mean_vector = np.mean(embeddings, axis=0)
    euclidean_dists = np.linalg.norm(embeddings - mean_vector, axis=1)
    threshold = np.percentile(euclidean_dists, distance_percent)  # top 1%
    farthest_outliers = euclidean_dists > threshold
    farthest_indices = indices[farthest_outliers]  # Record indices

    # === LOF outliers ===
    lof = LocalOutlierFactor(n_neighbors=lof_n_neighbors)
    lof_labels = lof.fit_predict(embeddings)  # -1 = outlier
    lof_outliers = lof_labels == -1
    lof_indices = indices[lof_outliers]  # Record indices

    # === Outlier dictionary ===
    outlier_info = {
        'farthest': {
            'indices': farthest_indices.tolist(),
            'distances': euclidean_dists[farthest_outliers].tolist(),
            'embeddings': embeddings[farthest_outliers].tolist()
        },
        'lof': {
            'indices': lof_indices.tolist(),
            'scores': lof.negative_outlier_factor_[lof_outliers].tolist(),
            'embeddings': embeddings[lof_outliers].tolist()
        },
        'both': {
            'indices': np.intersect1d(farthest_indices, lof_indices).tolist()
        }
    }

    return outlier_info

def plot_dbscan_k_distance(embeddings, min_samples=8, show=True, save_path=None, mark_elbow=True):
    """
    Plot the sorted k-distance (min_samples-th nearest neighbor distance) for DBSCAN eps selection,
    and optionally mark the elbow point.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        min_samples: int, usually set to DBSCAN's min_samples
        show: if True, display the plot
        save_path: if provided, save the plot to this path
        mark_elbow: if True, mark the elbow point on the plot
    """
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    k_distances = np.sort(distances[:, min_samples - 1])

    plt.figure()
    plt.plot(k_distances, label=f"{min_samples}-th NN distance")

    elbow_idx = None
    if mark_elbow:
        # Find the elbow using the "maximum distance to line" method
        n_points = len(k_distances)
        all_indices = np.arange(n_points)
        # Line from first to last point
        line_vec = np.array([n_points-1, k_distances[-1]]) - np.array([0, k_distances[0]])
        line_vec = line_vec / np.linalg.norm(line_vec)
        # Vector from first point to all points
        vec_from_first = np.vstack([all_indices, k_distances - k_distances[0]]).T
        # Compute the distance to the line
        line_normal = np.array([-line_vec[1], line_vec[0]])
        distances_to_line = np.abs(np.dot(vec_from_first, line_normal))
        elbow_idx = np.argmax(distances_to_line)
        plt.scatter(elbow_idx, k_distances[elbow_idx], color='red', label=f'Elbow (eps~{k_distances[elbow_idx]:.3f})')
        plt.axhline(k_distances[elbow_idx], color='red', linestyle='--', alpha=0.5)

    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{min_samples}-th NN distance")
    plt.title("k-distance plot for DBSCAN eps")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close()

    if mark_elbow:
        return k_distances[elbow_idx]
    else:
        return None


def find_boundaries(embeddings,
                    dbscan_eps=0.55,
                    dbscan_min_samples=10,
                    knn_n_neighbors=10,
                    knn_distance_percent=5):
    """
    Identify boundary samples in embedding space using DBSCAN and kNN density.
    Returns dictionary with boundary indices and metrics.
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
    
    Returns:
        boundary_info: dictionary containing:
            'dbscan_boundary': indices of DBSCAN boundary/noise points
            'low_density': indices of low-density points (bottom 5%)
            'both': indices meeting both criteria
            'density_scores': density values for all samples
            'dbscan_labels': full DBSCAN labels array
    """
    indices = np.arange(len(embeddings))

    # === DBSCAN boundary detection ===
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(embeddings)
    db_labels = db.labels_
    dbscan_boundary = db_labels == -1
    dbscan_boundary_indices = indices[dbscan_boundary]
    
    # === kNN Density Estimation ===
    knn = NearestNeighbors(n_neighbors=knn_n_neighbors)
    knn.fit(embeddings)
    distances, _ = knn.kneighbors(embeddings)
    density = 1 / np.mean(distances, axis=1)
    density_threshold = np.percentile(density, knn_distance_percent)
    low_density = density < density_threshold
    low_density_indices = indices[low_density]
    
    # === Find intersection ===
    both_indices = np.intersect1d(dbscan_boundary_indices, low_density_indices)
    
    # === Return boundary information ===
    boundary_info = {
        'dbscan_boundary': dbscan_boundary_indices.tolist(),
        'low_density': low_density_indices.tolist(),
        'both': both_indices.tolist(),
        'density_scores': density.tolist(),
        'dbscan_labels': db_labels.tolist()
    }
    
    return boundary_info


def summarize_analysis(embeddings, vulsamples_dir):
    with open(f"{vulsamples_dir}outlier_info.json", "r") as f:
        outlier_info = json.load(f)
    
    with open(f"{vulsamples_dir}boundary_info.json", "r") as f:
        boundary_info = json.load(f)
    
    # Step 3: Calculate counts
    summary = {
        'total_samples': len(embeddings),
        
        # Outlier counts
        'farthest_outliers': len(outlier_info['farthest']['indices']),
        'lof_outliers': len(outlier_info['lof']['indices']),
        'both_outliers': len(outlier_info['both']['indices']),
        'unique_outliers': len(set(outlier_info['farthest']['indices'] + 
                                outlier_info['lof']['indices'])),
        
        # Boundary counts
        'dbscan_boundaries': len(boundary_info['dbscan_boundary']),
        'low_density_boundaries': len(boundary_info['low_density']),
        'both_boundaries': len(boundary_info['both']),
        'unique_boundaries': len(set(boundary_info['dbscan_boundary'] + 
                                   boundary_info['low_density'])),
        
        # Overlap between boundaries and outliers
        'boundary_outlier_overlap': len(
            set(outlier_info['farthest']['indices'] + outlier_info['lof']['indices'])
            .intersection(
            set(boundary_info['dbscan_boundary'] + boundary_info['low_density'])
        ))
    }
    
    # Step 4: Calculate percentages
    summary['outlier_percentage'] = round(
        summary['unique_outliers'] / summary['total_samples'] * 100, 2
    )
    summary['boundary_percentage'] = round(
        summary['unique_boundaries'] / summary['total_samples'] * 100, 2
    )
    summary['overlap_percentage'] = round(
        summary['boundary_outlier_overlap'] / summary['total_samples'] * 100, 2
    )
    
    return summary


def save_outliers_and_boundaries(embeddings, 
                                 savepath, 
                                 outlier_path_prefix='outlier', 
                                 boundary_path_prefix='boundary',
                                 outlier_args=None,
                                 boundary_args=None):
    """
    Finds outliers and boundaries, then saves their indices (.npy) and full info (.json).
    """
    outlier_info = find_outliers(embeddings, **(outlier_args or {}))
    boundary_info = find_boundaries(embeddings, **(boundary_args or {}))

    # Save outlier indices and info
    outlier_indices = np.unique(outlier_info['farthest']['indices'])
    np.save(f'{savepath}/{outlier_path_prefix}_indices.npy', outlier_indices)
    with open(f'{savepath}/{outlier_path_prefix}_info.json', 'w') as f:
        json.dump(outlier_info, f)

    # Save boundary indices and info
    boundary_indices = np.unique(boundary_info['dbscan_boundary'] + boundary_info['low_density'])
    np.save(f'{savepath}/{boundary_path_prefix}_indices.npy', boundary_indices)
    with open(f'{savepath}/{boundary_path_prefix}_info.json', 'w') as f:
        json.dump(boundary_info, f)

    print(f"Saved outlier and boundary results to '{outlier_path_prefix}_*.npy/.json' and '{boundary_path_prefix}_*.npy/.json'.")


def split_dataset_for_private_and_aux(
    original_jsonl_path,
    special_indices,
    output_dir="split_datasets",
    ratio=2/3,
    prefix_p="D_p",
    prefix_a="D_a"
):
    """
    For an original dataset D in jsonl format:
      1. Exclude special samples using special_indices to get D'.
      2. Divide D' into D_p and D_a with size ratio=2:1 (D_p:D_a).
      3. Store D_p and D_a as jsonl files and their indices as .npy files.

    Args:
        original_jsonl_path: path to the original jsonl file
        special_indices: array-like of indices to exclude
        output_dir: directory to save the split datasets
        ratio: fraction of D' to use for D_p (default 2/3)
        prefix_p: prefix for private dataset files
        prefix_a: prefix for auxiliary dataset files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all original data into a list of dicts
    with open(original_jsonl_path, "r") as f:
        original_data = [json.loads(line) for line in f]

    all_indices = np.arange(len(original_data))
    special_indices = np.unique(special_indices)
    clean_indices = np.setdiff1d(all_indices, special_indices)

    # Shuffle clean indices
    np.random.shuffle(clean_indices)

    # Split according to ratio
    split_point = int(len(clean_indices) * ratio)
    indices_p = clean_indices[:split_point]
    indices_a = clean_indices[split_point:]

    # Save D_p
    np.save(os.path.join(output_dir, f"{prefix_p}_indices.npy"), indices_p)

    # Save D_a
    np.save(os.path.join(output_dir, f"{prefix_a}_indices.npy"), indices_a)

    print(f"Saved D_p ({len(indices_p)} samples) and D_a ({len(indices_a)} samples) to '{output_dir}'.")

def sample_target_datasets(
    special_indices,
    D_p_indices,
    original_jsonl_path,
    ori_embeddings_path,
    num_datasets=100,
    small_dataset_size=2500,
    output_dir="sampled_private_datasets"
):
    """
    Sample small private datasets and save their embeddings.

    Args:
        special_indices: 1D numpy array of special indices of vulnerable samples
        D_p_indices: 1D numpy array of indices of the private split D_p
        original_jsonl_path: path to the original JSONL file
        ori_embeddings_path: .npz with the embeddings of the original dataset
        num_datasets: number of small datasets to generate
        small_dataset_size: size of each small dataset (including special sample if added)
        output_dir: directory to save sampled datasets and labels
    """
    os.makedirs(output_dir, exist_ok=True)
    embeddings_save_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(embeddings_save_dir, exist_ok=True)

    data = np.load(ori_embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    additional_info = data["additional_info"]

    # Load all original data into a list of dicts
    with open(original_jsonl_path, "r") as f:
        original_data = [json.loads(line) for line in f]

    clean_indices = np.unique(D_p_indices)

    label_records = []

    for i in range(num_datasets):
        record = {
            "dataset_id": i,
            "added_special": False,
            "special_index": None,
            "special_text": None,
            "special_label": None,
            "not_added_samples": [],
            "sampled_indices": []
        }
        if random.random() < 0.5 and len(special_indices) > 0:
            # Add one special sample, sample the rest from clean
            sampled_clean = np.random.choice(clean_indices, size=small_dataset_size - 1, replace=False)
            special_idx = int(np.random.choice(special_indices))
            sampled_indices = np.append(sampled_clean, special_idx)
            record["added_special"] = True
            record["special_index"] = special_idx
            record["special_text"] = original_data[special_idx].get("text", None)
            record["special_label"] = original_data[special_idx].get("label", None)
        else:
            # No special sample, sample all from clean
            sampled_indices = np.random.choice(clean_indices, size=small_dataset_size, replace=False)
            # Randomly select 3 special samples for not_added_samples
            if len(special_indices) >= 3:
                chosen = np.random.choice(special_indices, size=3, replace=False)
            else:
                chosen = special_indices  # as many as possible
            for idx in chosen:
                idx = int(idx)
                record["not_added_samples"].append({
                    "index": idx,
                    "text": original_data[idx].get("text", None),
                    "label": original_data[idx].get("label", None)
                })
        sampled_indices = sampled_indices.astype(int)
        record["sampled_indices"] = sampled_indices.tolist()

        # Save sampled dataset in JSONL format
        jsonl_path = os.path.join(output_dir, f"dataset_{i}.jsonl")
        with open(jsonl_path, "w") as fout:
            for idx in sampled_indices:
                fout.write(json.dumps(original_data[idx], ensure_ascii=False) + "\n")

        # Save sampled embeddings as .npz
        sampled_embeddings = embeddings[sampled_indices]
        sampled_addtional_info = additional_info[sampled_indices]
        out_path = os.path.join(embeddings_save_dir, f"dataset_{i}_embeddings.npz")
        np.savez_compressed(out_path, embeddings=sampled_embeddings, additional_info=sampled_addtional_info)

        label_records.append(record)

    # Save label records as a JSON file
    with open(os.path.join(output_dir, "sampling_labels.json"), "w") as f:
        json.dump(label_records, f, indent=2, ensure_ascii=False)

    print(f"Generated {num_datasets} sampled datasets in '{output_dir}', saved sampling_labels.json, indices, and embeddings.")


def sample_Mref_datasets_with_specials(
    original_jsonl_path,
    original_embeddings_path,
    D_a_indices,
    special_indices,
    m,
    specialtype="outliers",
    output_dir="sampled_{m}ref_datasets_{specialtype}"
):
    """
    Given an original dataset in jsonl format and special indices (e.g., outliers),
    1. Extract samples which are not special but belonging to D_a_indices as the base dataset.
    2. Copy as M reference datasets.
    3. For each special sample, randomly select a half of the M datasets and add the special sample to them.
    4. Save each dataset as a jsonl file.
    5. Save a label file recording which special sample is added to which dataset, and its text and label.
    """
    output_dir = output_dir.format(m=m, specialtype=specialtype)
    os.makedirs(output_dir, exist_ok=True)

    # Load all original data into a list of dicts
    with open(original_jsonl_path, "r") as f:
        original_data = [json.loads(line) for line in f]

    special_indices = np.unique(special_indices)
    base_indices = np.setdiff1d(D_a_indices, special_indices) # to make sure base dataset does not contain special samples

    # Prepare m base datasets (as lists of indices)
    base_datasets = [list(base_indices.copy()) for _ in range(m)]

    # Record which special sample is added to which dataset
    label_records = []

    for special_idx in special_indices:
        # Randomly select two of the m datasets
        chosen_datasets = random.sample(range(m), 2)
        for ds_id in chosen_datasets:
            base_datasets[ds_id].append(int(special_idx))
        label_records.append({
            "special_index": int(special_idx),
            "special_text": original_data[special_idx].get("text", None),
            "special_label": original_data[special_idx].get("label", None),
            "added_to_datasets": chosen_datasets
        })

    embedding_output_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(embedding_output_dir, exist_ok=True)
    data = np.load(original_embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    additional_info = data["additional_info"]

    # Save each dataset as a jsonl file and indices as .npy
    for i, indices in enumerate(base_datasets):
        indices = np.array(indices, dtype=int)
        jsonl_path = os.path.join(output_dir, f"dataset_{i}.jsonl")
        with open(jsonl_path, "w") as fout:
            for idx in indices:
                fout.write(json.dumps(original_data[idx], ensure_ascii=False) + "\n")
        sampled_embeddings = embeddings[indices]
        sampled_additional_info = additional_info[indices]
        out_path = os.path.join(embedding_output_dir, f"dataset_{i}_embeddings.npz")
        np.savez_compressed(out_path, embeddings=sampled_embeddings, additional_info=sampled_additional_info)
        
    # Save label records as a JSON file
    with open(os.path.join(output_dir, "sampling_labels.json"), "w") as f:
        json.dump(label_records, f, indent=2, ensure_ascii=False)

    print(f"Generated {m} base datasets in '{output_dir}', with special samples distributed and labels saved.")
    print(f"Saved sampled embeddings (npz) for {m} datasets to '{embedding_output_dir}'.")


def main(datasetname):
    """Build the MIA data of one dataset under data/<dataset>/mia/ (run from the repo root)."""
    datarootdir = f"./data/{datasetname}/"
    original_jsonl_path = f"{datarootdir}original/train-original.jsonl"
    embed_dir = f"{datarootdir}mia/embeds/"
    original_embeddings_path = f"{embed_dir}cls_{datasetname}_train_all.embeddings.npz"
    specialsavepath = f'{datarootdir}mia/vulsamples/'
    specialconfig_path = f"{specialsavepath}config.json"
    sampling_output_dir = f'{datarootdir}mia/D_sample/'
    private_indices_path = f"{sampling_output_dir}split_datasets/D_p_indices.npy"
    reference_indices_path = f"{sampling_output_dir}split_datasets/D_a_indices.npy"

    # Step 1: Compute and save embeddings for the original dataset (if not already done)
    print("Step 1: Computing embeddings for the original dataset...")
    compute_embeddings(datasetname, original_jsonl_path, embed_dir, model_name="sentence-t5-base", batch_size=200)

    # step 2: Load embeddings and analyze outliers and boundaries, then save results
    print("Step 2: Analyzing embeddings to find outliers and boundaries...")
    data = np.load(original_embeddings_path)
    embeddings = data["embeddings"]
    plot_dbscan_k_distance(embeddings, min_samples=5, show=False, save_path=f"./outputs/figs/dbscan_k_distance_{datasetname}.png")

    # step 3: data/<dataset>/mia/vulsamples/config.json holds the parameters of find_outliers and find_boundaries, e.g.:
    # {
    #     "find_outliers": {
    #         "distance_percent": 99,
    #         "lof_n_neighbors": 10
    #     },
    #     "find_boundaries": {
    #         "dbscan_eps": 0.55,
    #         "dbscan_min_samples": 10,
    #         "knn_n_neighbors": 10,
    #         "knn_distance_percent": 5
    #     }
    # }
    print("Step 3: Finding outliers and boundaries, and saving results...")
    with open(specialconfig_path, "r") as f:
        config = json.load(f)
    outlier_args = config.get("find_outliers", {})
    boundary_args = config.get("find_boundaries", {})
    save_outliers_and_boundaries(embeddings, savepath=specialsavepath, 
                                 outlier_args=outlier_args, boundary_args=boundary_args)

    summary = summarize_analysis(embeddings, specialsavepath)
    print("Analysis Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Step 4: Load special indices and split dataset, sample target datasets, and sample Mref datasets with specials
    print("Step 4: Loading special indices, splitting dataset, and sampling datasets for D_p and D_a...")
    special_indices = np.load(f'{specialsavepath}/outlier_indices.npy')
    specialtype = "outliers"
    print("Number of outliers:", len(special_indices))

    split_dataset_for_private_and_aux(
        original_jsonl_path,
        special_indices,
        output_dir=f"{sampling_output_dir}split_datasets",
        ratio=2/3,
        prefix_p="D_p",
        prefix_a="D_a"
    )
    # After this step, you will have D_p_indices.npy and D_a_indices.npy in the split_datasets directory.

    # Step 5: Sample target datasets for D_p
    print("Step 5: Sampling target datasets for D_p and Mref datasets with specials for D_a...")
    D_p_indices = np.load(private_indices_path)
    print("The shape of D_p_indices is:", D_p_indices.shape)
    sample_target_datasets(
        special_indices,
        D_p_indices,
        original_jsonl_path,
        original_embeddings_path,
        num_datasets=100,
        small_dataset_size=1000,
        output_dir=f"{sampling_output_dir}sampled_private_datasets"
    )

    # Step 6: Sample M reference datasets with specials from D_a
    print("Step 6: Sampling Mref datasets with specials for D_a...")
    D_a_indices = np.load(reference_indices_path)
    print("The shape of D_a_indices is:", D_a_indices.shape)
    m = 4
    sample_Mref_datasets_with_specials(
        original_jsonl_path,
        original_embeddings_path,
        D_a_indices,
        special_indices,
        m,
        specialtype="outliers",
        output_dir=f"{sampling_output_dir}sampled_{m}ref_datasets_{specialtype}"
    )

    # After running this code, you will have:
    # 1. A set of sampled private datasets in "sampled_private_datasets" directory, each with a JSONL file and corresponding embeddings in .npz format.
    # 2. A set of sampled reference datasets with special samples in "sampled_{m}ref_datasets_{specialtype}" directory, each with a JSONL file and corresponding embeddings in .npz format.


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "DMSAFN")
