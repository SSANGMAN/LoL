class ExplainModel:
    def __init__(self, model, col_names):
        self.model = model
        self.col_names = col_names

    def FeatureImportance(self, importance_type):
        if str(type(self.model)).split('.')[-1][:-2] == 'XGBClassifier':
            model = self.model
            model.get_booster().feature_names = self.col_names
            xgb.plot_importance(model,importance_type = importance_type, max_num_features = 5)

            plt.title("Feature Importance ({})".format(importance_type))
            plt.show()
            
        elif str(type(self.model)).split('.')[-1][:-2] == 'RandomForestClassifier':
            imp_df = pd.DataFrame({'Feature' : self.col_names, 'Importance' : self.model.feature_importances_})
            top5_imp_df = imp_df.sort_values(by = 'Importance', ascending = False).iloc[:5]
            sns.barplot(x = 'Importance', y = 'Feature', data = top5_imp_df)
            
            plt.title('RandomForest Feature Importances (based on Gini index)')
            plt.show
            
    def SHAP_Value(self, test_data, unit, index = 10, method = 'tree'):
        if unit == 'individual':
            test_idx = np.array([test_data[index]])

            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(test_idx)
            
            shap.initjs()
            
            if str(type(self.model)).split('.')[-1][:-2] == 'XGBClassifier':
                return shap.force_plot(explainer.expected_value, shap_values[0, :], test_idx, link = 'logit', feature_names = self.col_names)
            
            elif str(type(self.model)).split('.')[-1][:-2] == 'RandomForestClassifier':
                return shap.force_plot(explainer.expected_value[1], shap_values[1], test_idx, feature_names = self.col_names)