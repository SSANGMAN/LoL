class ExplainModel:
    def __init__(self, model, col_names):
        self.model = model
        self.model_dim = len(self.model.get_booster().feature_names)
        self.col_names = col_names

    def FeatureImportance(self, importance_type):
        if str(type(self.model)).split('.')[-1][:-2] == 'XGBClassifier':
            model = self.model
            model.get_booster().feature_names = self.col_names
            xgb.plot_importance(
                self.model, importance_type=importance_type, max_num_features=5)

            plt.title("Feature Importance ({})".format(importance_type))
            plt.show()

        elif str(type(self.model)).split('.')[-1][:-2] == 'RandomForestClassifier':
            imp_df = pd.DataFrame(
                {'Feature': self.col_names, 'Importance': self.model.feature_importances_})
            top5_imp_df = imp_df.sort_values(
                by='Importance', ascending=False).iloc[:5]
            sns.barplot(x='Importance', y='Feature', data=top5_imp_df)

            plt.title('RandomForest Feature Importances (based on Gini index)')
            plt.show

    def SHAP_Value(self, test_data, unit, index=10, method='tree'):
        explainer = shap.TreeExplainer(self.model)

        if unit == 'force':
            test_idx = np.array([test_data[index]])
            shap_values = explainer.shap_values(test_idx)

            shap.initjs()

            if str(type(self.model)).split('.')[-1][:-2] == 'XGBClassifier':
                return shap.force_plot(explainer.expected_value, shap_values[0, :], test_idx, link='logit', feature_names=self.col_names)

            elif str(type(self.model)).split('.')[-1][:-2] == 'RandomForestClassifier':
                return shap.force_plot(explainer.expected_value[1], shap_values[1], test_idx, feature_names=self.col_names)

        elif unit == 'summary':
            shap_values = explainer.shap_values(test_data)
            shap.initjs()

            return shap.summary_plot(shap_values, test_data, plot_type='bar', feature_names=self.col_names)

    def PartialDependencePlot(self, train_data, feature_names):
        default_model_col_names = ["f{}".format(
            i) for i in range(self.model_dim)]
        self.model.get_booster().feature_names = default_model_col_names

        feature_index_list = [
            feature for feature in range(train_data.shape[1])]
        feature_dict = dict(zip(self.col_names, feature_index_list))
        selected_features_idx = [feature_dict[feature_name]
                                 for feature_name in feature_names]

        plot_partial_dependence(
            self.model, train_data, features=selected_features_idx, feature_names=self.col_names)

        plt.suptitle('Partial Dependence Plot')
        plt.show()
