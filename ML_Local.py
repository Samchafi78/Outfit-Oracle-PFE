import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from VGG16_V2 import MultiOutputVGG
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
import sys
import os
import json


def run_model(image_path):
    print(f"Traitement de l'image: {image_path}")

    # --- Configuration --- #
    checkpoint_path = os.path.join(os.path.dirname(__file__), "best_model.pth")

    # Device define
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used is : {device}")

    # --- Category-specific indices --- #
    shape_indices = {
        "upper": [0, 8, 9, 11],
        "lower": [1],
        "outer": [10],
        "other": [i for i in range(12) if i not in [0, 1, 8, 9, 10, 11]]
    }
    fabric_indices = {
        "upper": [0],
        "lower": [1],
        "outer": [2]
    }
    color_indices = {
        "upper": [0],
        "lower": [1],
        "outer": [2]
    }

    # --- Load model --- #
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    num_ftrs = vgg.classifier[0].in_features
    model = MultiOutputVGG(vgg, num_ftrs).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    thresholds = checkpoint.get("thresholds")
    shape_thresh = np.array(thresholds["shape"])
    fabric_thresh = np.array(thresholds["fabric"])
    pattern_thresh = np.array(thresholds["pattern"])
    model.eval()

    # --- Image Preprocessing --- #
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # --- Run Inference --- #
    with torch.no_grad():
        shape_out, fabric_out, pattern_out = model(image_tensor)
        shape_prob = torch.sigmoid(shape_out).cpu().numpy()[0]
        fabric_prob = torch.sigmoid(fabric_out).cpu().numpy()[0]
        pattern_prob = torch.sigmoid(pattern_out).cpu().numpy()[0]

        shape_binary = (shape_prob > shape_thresh).astype(int)
        fabric_binary = (fabric_prob > fabric_thresh).astype(int)
        pattern_binary = (pattern_prob > pattern_thresh).astype(int)


    # --- Output per category --- #
    shape_dims = [6, 5, 4, 3, 5, 3, 3, 3, 5, 7, 3, 3]
    offset = 0
    user_upper=[]
    user_lower=[]
    user_outer=[]
    user_other=[]

    for category in ["upper", "lower", "outer", "other"]:
        shape_idxs = shape_indices[category]
        fabric_idx = fabric_indices[category][0] if category != "other" else None
        color_idx = color_indices[category][0] if category != "other" else None

        # Extract shape
        shape_selected = []
        offset = 0
        for j in range(12):
            attr = shape_binary[offset:offset+shape_dims[j]]
            if j in shape_idxs:
                shape_selected.extend(attr)
            offset += shape_dims[j]

        # Fabric/Color
        fabric_label = list(map(int, fabric_binary[fabric_idx*8:(fabric_idx+1)*8])) if fabric_idx is not None else []
        color_label = list(map(int, pattern_binary[color_idx*8:(color_idx+1)*8])) if color_idx is not None else []

        if category=="upper":
            user_upper.append({
            "shape_label": shape_selected,  # Directly as a list of integers
            "fabric_label": fabric_label,   # List of integers
            "color_label": color_label     # List of integers
            })

        if category=="lower":
            user_lower.append({
            "shape_label": shape_selected,  # Directly as a list of integers
            "fabric_label": fabric_label,   # List of integers
            "color_label": color_label     # List of integers
            })

        if category=="outer":
            user_outer.append({
            "shape_label": shape_selected,  # Directly as a list of integers
            "fabric_label": fabric_label,   # List of integers
            "color_label": color_label     # List of integers
            })

    # --- Finding most similar item --- #
    # Load the article features from CSV
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_hm", "hm_inference_results.csv"))
    # Extract integer from tensor strings
    df['article_id'] = df['article_id'].str.extract(r'tensor\((\d+)\)').astype(int)
    # Check results
    #print(df.head())

    categories = ['upper', 'lower', 'outer', 'other']
    # Split into 4 DataFrames by category
    df_upper = df[df['category'] == 'upper']
    df_lower = df[df['category'] == 'lower']
    df_outer = df[df['category'] == 'outer']

    # Check results
    #print(df_outer.head(10))


    # Helper function to flatten user feature dictionary into a single vector
    def flatten_features(entry):
        return entry["shape_label"] + entry["fabric_label"] + entry["color_label"]

    # Convert user upper vector to NumPy array
    user_vector_upper = np.array(flatten_features(user_upper[0])).reshape(1, -1)
    if df_upper is not None :
        # Convert DataFrame features to list of vectors
        df_upper_vectors = df_upper[["shape_label", "fabric_label", "color_label"]].apply(
            lambda row: flatten_features({
                "shape_label": list(map(int, row["shape_label"].split(","))),
                "fabric_label": list(map(int, row["fabric_label"].split(","))),
                "color_label": list(map(int, row["color_label"].split(",")))
            }), axis=1
        )
        df_upper_matrix = np.stack(df_upper_vectors.to_numpy())

        # Compute cosine similarity
        upper_similarities = cosine_similarity(user_vector_upper, df_upper_matrix)[0]
        best_match_idx = np.argmax(upper_similarities)

        # Get the best article ID
        best_upper_article_id = df_upper.iloc[best_match_idx]["article_id"]
        print("Most similar upper article ID from H&M dataset:", best_upper_article_id)

    user_vector_lower = np.array(flatten_features(user_lower[0])).reshape(1, -1)
    if df_lower is not None :
        # Convert DataFrame features to list of vectors
        df_lower_vectors = df_lower[["shape_label", "fabric_label", "color_label"]].apply(
            lambda row: flatten_features({
                "shape_label": list(map(int, row["shape_label"].split(","))),
                "fabric_label": list(map(int, row["fabric_label"].split(","))),
                "color_label": list(map(int, row["color_label"].split(",")))
            }), axis=1
        )
        df_lower_matrix = np.stack(df_lower_vectors.to_numpy())

        # Compute cosine similarity
        lower_similarities = cosine_similarity(user_vector_lower, df_lower_matrix)[0]
        best_match_idx = np.argmax(lower_similarities)

        # Get the best article ID
        best_lower_article_id = df_lower.iloc[best_match_idx]["article_id"]
        print("Most similar lower article ID from H&M dataset:", best_lower_article_id)

    user_vector_outer = np.array(flatten_features(user_outer[0])).reshape(1, -1)
    if df_outer is not None :
        # Convert DataFrame features to list of vectors
        df_outer_vectors = df_outer[["shape_label", "fabric_label", "color_label"]].apply(
            lambda row: flatten_features({
                "shape_label": list(map(int, row["shape_label"].split(","))),
                "fabric_label": list(map(int, row["fabric_label"].split(","))),
                "color_label": list(map(int, row["color_label"].split(",")))
            }), axis=1
        )
        df_outer_matrix = np.stack(df_outer_vectors.to_numpy())

        # Compute cosine similarity
        outer_similarities = cosine_similarity(user_vector_outer, df_outer_matrix)[0]
        best_match_idx = np.argmax(outer_similarities)

        # Get the best article ID
        best_outer_article_id = df_outer.iloc[best_match_idx]["article_id"]
        print("Most similar outer article ID from H&M dataset:", best_outer_article_id)


    # --- UUCF --- #
    # Load transactions data
    transactions = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_hm","transactions_train.csv"))
    # Check results
    #print(transactions.head(10))

    # Convert customer IDs into a categorical data type and Extract numerical codes (integer labels) assigned by pandas
    transactions["user_index"] = transactions["customer_id"].astype("category").cat.codes
    transactions["item_index"] = transactions["article_id"].astype("category").cat.codes

    # Now build the user-item sparse matrix (each purchase gets a 1)
    user_item_matrix = coo_matrix(
        (np.ones(len(transactions)), (transactions["user_index"], transactions["item_index"]))
    )

    # Fit kNN model on user-item matrix
    knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    knn_model.fit(user_item_matrix)

    #Simulate a new user who matched article_id = best_article_id
    best_article_ids = {
        "upper": best_upper_article_id,
        "lower": best_lower_article_id,
        "outer": best_outer_article_id
    }
    final_recommendations = set()
    for cat in ['upper', 'lower', 'outer']:
        best_article_id = best_article_ids[cat]
        print(f"Best {cat} article ID:", best_article_id)

        # Find users who purchased this article
        customers_who_bought = transactions[transactions["article_id"] == best_article_id]["customer_id"].unique()

        #Find similar article by using UUCF
        if len(customers_who_bought) > 0:
            # Take one of these customers as a reference
            reference_customer = customers_who_bought[0]

            # Correct indexing into sparse matrix
            reference_user_index = transactions.loc[
                transactions["customer_id"] == reference_customer, "user_index"
            ].iloc[0]

            reference_vector = user_item_matrix.getrow(reference_user_index)

            # Find k nearest users to the reference
            distances, indices = knn_model.kneighbors(reference_vector, n_neighbors=10)

            # Collect article purchases from similar users
            similar_user_indices = indices.flatten()
            similar_user_ids = transactions.loc[
                transactions["user_index"].isin(similar_user_indices), "customer_id"
            ].unique()

            # Get top recommended articles (excluding the one already matched)
            similar_user_purchases = transactions[
                transactions["customer_id"].isin(similar_user_ids)
            ]

            recommended_articles = (
                similar_user_purchases[similar_user_purchases["article_id"] != best_article_id]
                .article_id.value_counts()
                .head(5)
            )
            print("🔁 Recommended Articles from Similar Users:")
            print(recommended_articles)
            # Ajouter le premier article de chaque catégorie
            for article_id in recommended_articles.index:
                if article_id != best_article_id:
                    final_recommendations.add(str(article_id))
                    break  # ajoute un seul article par catégorie
        else:
            print(f"No customers found for article_id: {best_article_id}")
    # Compléter à 5 articles si besoin
    if len(final_recommendations) < 5:
        top_articles = (
            transactions.article_id.value_counts()
            .loc[~transactions.article_id.value_counts().index.isin(final_recommendations)]
            .head(5 - len(final_recommendations))
        )
        final_recommendations.update(top_articles.index.astype(str))

    results = {
        "best_articles": {
            "upper": str(best_article_ids["upper"]).zfill(10),
            "lower": str(best_article_ids["lower"]).zfill(10),
            "outer": str(best_article_ids["outer"]).zfill(10),
        },
        "recommendations": [str(aid).zfill(10) for aid in final_recommendations]
    }
    print("🧪 Articles recommandés :", recommended_articles.index.tolist())

    # Sauvegarder le JSON dans le dossier PFE_model/output/
    output_path = os.path.join(os.path.dirname(__file__), "output", "results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f)

    print("✅ Résultats sauvegardés dans results.json")
    


if __name__ == "__main__":
    image_path = sys.argv[1]
    run_model(image_path)