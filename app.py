import gradio as gr
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

def plot_feature_importance():
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    
    filepath = 'plots/feature_importance.png'
    plt.savefig(filepath)
    plt.close()
    
    return filepath

def plot_input_summary(square_feet, bedrooms, bathrooms, age_years, lot_size, garage_spaces, neighborhood_score):
    values = [square_feet, bedrooms, bathrooms, age_years, lot_size, garage_spaces, neighborhood_score]
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, values, color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Valeurs')
    plt.title('Valeurs d\'entrée de l\'utilisateur')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filepath = 'plots/input_summary.png'
    plt.savefig(filepath)
    plt.close()
    
    return filepath

def plot_prediction(prediction, std_dev):
    lower_bound = prediction - 1.96 * std_dev
    upper_bound = prediction + 1.96 * std_dev
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Prédiction'], [prediction], color='green', alpha=0.7, label='Prédiction')
    plt.errorbar(['Prédiction'], [prediction], yerr=[1.96 * std_dev], 
                 fmt='o', color='black', capsize=10, label='Intervalle de confiance 95%')
    plt.ylabel('Prix ($)')
    plt.title('Prédiction du prix avec intervalle de confiance')
    plt.legend()
    plt.tight_layout()
    
    filepath = 'plots/prediction.png'
    plt.savefig(filepath)
    plt.close()
    
    return filepath

def predict_price(square_feet, bedrooms, bathrooms, age_years, lot_size, garage_spaces, neighborhood_score):
    # Créer un DataFrame avec les noms de colonnes pour éviter l'avertissement
    input_data = pd.DataFrame([[square_feet, bedrooms, bathrooms, age_years, lot_size, garage_spaces, neighborhood_score]], 
                              columns=feature_names)
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    
    tree_predictions = np.array([tree.predict(input_scaled)[0] for tree in model.estimators_])
    std_dev = np.std(tree_predictions)
    
    lower_bound = prediction - 1.96 * std_dev
    upper_bound = prediction + 1.96 * std_dev
    
    feature_importance_plot = plot_feature_importance()
    input_summary_plot = plot_input_summary(square_feet, bedrooms, bathrooms, age_years, lot_size, garage_spaces, neighborhood_score)
    prediction_plot = plot_prediction(prediction, std_dev)
    
    result_text = f"""
    Prix prédit: ${prediction:,.2f}
    
    Intervalle de confiance à 95%:
    - Limite inférieure: ${lower_bound:,.2f}
    - Limite supérieure: ${upper_bound:,.2f}
    """
    
    return feature_importance_plot, input_summary_plot, prediction_plot, result_text

with gr.Blocks() as demo:
    gr.Markdown("# 🏠 Prédicteur de Prix Immobiliers")
    gr.Markdown("Ajustez les caractéristiques de la maison pour prédire son prix")
    
    with gr.Row():
        with gr.Column():
            square_feet = gr.Slider(minimum=500, maximum=5000, value=2000, label="Surface (pieds carrés)")
            bedrooms = gr.Slider(minimum=1, maximum=6, value=3, step=1, label="Nombre de chambres")
            bathrooms = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Nombre de salles de bain")
            age_years = gr.Slider(minimum=0, maximum=50, value=10, label="Âge (années)")
            lot_size = gr.Slider(minimum=1000, maximum=20000, value=5000, label="Taille du terrain (pieds carrés)")
            garage_spaces = gr.Slider(minimum=0, maximum=3, value=2, step=1, label="Places de garage")
            neighborhood_score = gr.Slider(minimum=1, maximum=10, value=7, label="Score du quartier")
            
            predict_btn = gr.Button("Prédire le prix")
        
        with gr.Column():
            result_text = gr.Textbox(label="Résultat de la prédiction", lines=5)
            
            with gr.Row():
                feature_importance_img = gr.Image(label="Importance des Features", type="filepath")
            
            with gr.Row():
                input_summary_img = gr.Image(label="Résumé des Entrées", type="filepath")
            
            with gr.Row():
                prediction_img = gr.Image(label="Prédiction avec Intervalle de Confiance", type="filepath")
    
    predict_btn.click(
        fn=predict_price,
        inputs=[square_feet, bedrooms, bathrooms, age_years, lot_size, garage_spaces, neighborhood_score],
        outputs=[feature_importance_img, input_summary_img, prediction_img, result_text]
    )

demo.launch()