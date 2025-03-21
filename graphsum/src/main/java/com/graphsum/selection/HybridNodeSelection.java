package com.graphsum.selection;

import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.type.StructType;
//LinearModel For Linear Regression, Bayesian Ridge, ElasticNet
//import smile.regression.ElasticNet;
import smile.regression.*;
import smile.classification.AdaBoost;
import smile.classification.DecisionTree;
import smile.feature.extraction.PCA;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.*;

public class HybridNodeSelection {
    private final int k;
    private final double percentage;
    private final boolean useEmbeddings;
    private final String embeddingsPath;
    private final double alpha; // Weight for structural vs semantic importance
    private Map<String, double[]> embeddings;
    private final boolean usePercentage; // Flag to determine if we're using percentage or k
    private Map<String, Integer> nodeAccessCounts;
    private Map<String, Integer> edgeAccessCounts;

    private static class ModelWrapper {
        String identifier;
        Object model;
    
        public ModelWrapper(String identifier, Object model) {
            this.identifier = identifier;
            this.model = model;
        }
    }

    public HybridNodeSelection(int k, boolean useEmbeddings, String embeddingsPath) {
        this(k, useEmbeddings, embeddingsPath, 0.7); // Default alpha = 0.7
    }
    
    public HybridNodeSelection(int k, boolean useEmbeddings, String embeddingsPath, double alpha) {
        this.k = k;
        this.percentage = -1; // Not using percentage
        this.usePercentage = false;
        this.useEmbeddings = useEmbeddings;
        this.embeddingsPath = embeddingsPath;
        this.alpha = alpha;
        
        if (useEmbeddings && embeddingsPath != null) {
            this.embeddings = loadEmbeddings(embeddingsPath);
        }
    }
    
    // New constructor for percentage-based selection
    public HybridNodeSelection(double percentage, boolean useEmbeddings, String embeddingsPath) {
        this(percentage, useEmbeddings, embeddingsPath, 0.7); // Default alpha = 0.7
    }
    
    // New constructor for percentage-based selection with alpha
    public HybridNodeSelection(double percentage, boolean useEmbeddings, String embeddingsPath, double alpha) {
        if (percentage <= 0 || percentage > 100) {
            throw new IllegalArgumentException("Percentage must be between 0 and 100");
        }
        this.percentage = percentage;
        this.k = -1; // Not using fixed k
        this.usePercentage = true;
        this.useEmbeddings = useEmbeddings;
        this.embeddingsPath = embeddingsPath;
        this.alpha = alpha;
        
        if (useEmbeddings && embeddingsPath != null) {
            this.embeddings = loadEmbeddings(embeddingsPath);
        }
    }
    
    // Load embeddings from file
    private Map<String, double[]> loadEmbeddings(String path) {
        Map<String, double[]> embeddings = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line;
            // Skip header if exists
            if ((line = reader.readLine()) != null && line.startsWith("entity")) {
                // Header line, skip
            }
            
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split("\\s+");
                if (parts.length > 2) {
                    String entityId = parts[0];
                    // Remove http://www.wikidata.org/entity/ prefix if present
                    entityId = entityId.replace("http://www.wikidata.org/entity/", "");
                    
                    double[] vector = new double[parts.length - 1];
                    for (int i = 1; i < parts.length; i++) {
                        vector[i-1] = Double.parseDouble(parts[i]);
                    }
                    embeddings.put(entityId, vector);
                }
            }
            System.out.printf("Node selection loaded %d embeddings with dimension %d%n", 
                embeddings.size(), 
                embeddings.isEmpty() ? 0 : embeddings.values().iterator().next().length);
        } catch (IOException e) {
            System.err.println("Error loading embeddings: " + e.getMessage());
        }
        return embeddings;
    }

    /**
     * Loads query log data to be used as features for ML training
     * 
     * @param nodeAccessCounts Map of node IDs to their access counts from query logs
     * @param edgeAccessCounts Map of edge keys to their access counts from query logs
     */
    public void loadQueryLogFeatures(Map<String, Integer> nodeAccessCounts, 
                                    Map<String, Integer> edgeAccessCounts) {
        this.nodeAccessCounts = new HashMap<>(nodeAccessCounts);
        this.edgeAccessCounts = new HashMap<>(edgeAccessCounts);
        System.out.println("Loaded query log features: " + nodeAccessCounts.size() + 
                " node access records, " + edgeAccessCounts.size() + " edge access records");
    }

    public List<String> selectTopKNodes(Map<String, Map<String, Double>> centralityScores) 
            throws InstantiationException, IllegalAccessException, InvocationTargetException {
        // Convert centrality scores into a DataFrame for ML processing
        List<String> nodeIds = new ArrayList<>(centralityScores.values().iterator().next().keySet());
        
        // Calculate k if using percentage
        int effectiveK = this.k;
        if (usePercentage) {
            effectiveK = (int) Math.ceil((percentage / 100.0) * nodeIds.size());
            System.out.println("Using percentage-based selection: " + percentage + "% of nodes");
            System.out.println("Total nodes: " + nodeIds.size() + ", selecting top-" + effectiveK + " nodes");
        }

        System.out.println("Selecting top-" + effectiveK + " nodes using " + 
                (useEmbeddings ? "hybrid structural-semantic" : "structural") + " approach...");

        // Extract structural features from centrality scores
        List<String> structuralMeasures = new ArrayList<>(centralityScores.keySet());
        // Filter out semantic measures if they exist in the centrality scores
        structuralMeasures.removeIf(measure -> 
            measure.equals("semanticPageRank") || measure.equals("embeddingDiversity"));
        
        int structuralFeatureCount = structuralMeasures.size();
        
        // Build the feature matrix
        double[][] structuralFeatures = new double[nodeIds.size()][structuralFeatureCount];
        
        // Fill in structural features
        for (int i = 0; i < structuralMeasures.size(); i++) {
            String measure = structuralMeasures.get(i);
            Map<String, Double> scores = centralityScores.get(measure);
            for (int j = 0; j < nodeIds.size(); j++) {
                structuralFeatures[j][i] = scores.getOrDefault(nodeIds.get(j), 0.0);
            }
        }
        
        // Create initial column names for the DataFrame
        String[] structuralColumnNames = structuralMeasures.toArray(new String[0]);
        
        // Create DataFrame for structural features
        DataFrame structuralDf = DataFrame.of(structuralFeatures, structuralColumnNames);
        
        // Add query log features if available
        if (nodeAccessCounts != null && !nodeAccessCounts.isEmpty()) {
            System.out.println("Adding query log features to ML training data");
            
            // Create query frequency features
            double[] queryFrequencies = new double[nodeIds.size()];
            for (int i = 0; i < nodeIds.size(); i++) {
                queryFrequencies[i] = nodeAccessCounts.getOrDefault(nodeIds.get(i), 0);
            }
            
            // Add edge-based features
            double[] edgeFrequencies = new double[nodeIds.size()];
            if (edgeAccessCounts != null && !edgeAccessCounts.isEmpty()) {
                for (int i = 0; i < nodeIds.size(); i++) {
                    String nodeId = nodeIds.get(i);
                    double edgeCount = 0;
                    
                    // Count edge accesses involving this node
                    for (Map.Entry<String, Integer> entry : edgeAccessCounts.entrySet()) {
                        String edgeKey = entry.getKey();
                        if (edgeKey.contains(nodeId)) {
                            edgeCount += entry.getValue();
                        }
                    }
                    
                    edgeFrequencies[i] = edgeCount;
                }
            }
            
            // Create DataFrames for query features
            DataFrame queryDf = DataFrame.of(transformTo2D(queryFrequencies), new String[]{"query_frequency"});
            DataFrame edgeDf = DataFrame.of(transformTo2D(edgeFrequencies), new String[]{"edge_frequency"});
            
            // Merge with structural features
            structuralDf = structuralDf.merge(queryDf).merge(edgeDf);
        }
        
        // Create target variable Y
        double[] y = new double[nodeIds.size()];
        
        // If embeddings are available, compute hybrid importance scores
        if (useEmbeddings && embeddings != null && !embeddings.isEmpty()) {
            // Calculate semantic importance and combine with structural
            y = computeHybridImportanceScore(centralityScores, nodeIds);
            
            // Add semantic features if available in centrality scores
            List<String> semanticMeasures = Arrays.asList("semanticPageRank", "embeddingDiversity");
            List<String> availableSemantic = new ArrayList<>();
            
            for (String measure : semanticMeasures) {
                if (centralityScores.containsKey(measure)) {
                    availableSemantic.add(measure);
                }
            }
            
            if (!availableSemantic.isEmpty()) {
                double[][] semanticFeatures = new double[nodeIds.size()][availableSemantic.size()];
                
                for (int i = 0; i < availableSemantic.size(); i++) {
                    String measure = availableSemantic.get(i);
                    Map<String, Double> scores = centralityScores.get(measure);
                    for (int j = 0; j < nodeIds.size(); j++) {
                        semanticFeatures[j][i] = scores.getOrDefault(nodeIds.get(j), 0.0);
                    }
                }
                
                // Create DataFrame for semantic features
                DataFrame semanticDf = DataFrame.of(semanticFeatures, 
                                                   availableSemantic.toArray(new String[0]));
                
                // Merge with structural features
                structuralDf = structuralDf.merge(semanticDf);
            }
            
            // If we have embeddings for at least some nodes, add embedding features
            int embeddingCount = 0;
            for (String nodeId : nodeIds) {
                if (embeddings.containsKey(nodeId)) {
                    embeddingCount++;
                }
            }
            
            if (embeddingCount > 0) {
                // Get embedding dimension
                int embeddingDim = embeddings.values().iterator().next().length;
                System.out.println("Adding embedding features (dimension: " + embeddingDim + ")");
                
                // For high-dimensional embeddings, use PCA to reduce dimensions
                int targetDim = Math.min(embeddingDim, 10); // Limit to 10 dimensions
                
                // Create matrix for embeddings
                double[][] embeddingFeatures = new double[nodeIds.size()][embeddingDim];
                
                // Fill embedding features
                for (int i = 0; i < nodeIds.size(); i++) {
                    String nodeId = nodeIds.get(i);
                    if (embeddings.containsKey(nodeId)) {
                        double[] nodeEmbedding = embeddings.get(nodeId);
                        System.arraycopy(nodeEmbedding, 0, embeddingFeatures[i], 0, embeddingDim);
                    } else {
                        // Use zeros for missing embeddings
                        Arrays.fill(embeddingFeatures[i], 0.0);
                    }
                }
                
                // Create embedding column names
                String[] embeddingColumnNames = new String[embeddingDim];
                for (int i = 0; i < embeddingDim; i++) {
                    embeddingColumnNames[i] = "emb_" + i;
                }
                
                // Create DataFrame for embeddings
                DataFrame embeddingDf = DataFrame.of(embeddingFeatures, embeddingColumnNames);
                
                // Merge with structural features
                structuralDf = structuralDf.merge(embeddingDf);
            }
        } else {
            // If no embeddings, use structural importance only
            y = computeHeuristicScore(centralityScores, nodeIds);
        }
        
        System.out.println("Feature matrix shape: " + structuralDf.size() + " x " + structuralDf.ncol());
        
        // Add target variable to DataFrame
        DataFrame yDf = DataFrame.of(transformTo2D(y), new String[]{"Y"});
        DataFrame fullDf = structuralDf.merge(yDf);
        
        // Train regression models
        ModelWrapper bestModelWrapper = trainBestRegressor(fullDf, "Y");
        Object bestModel = bestModelWrapper.model;
        
        // Make predictions with the best model
        double[] predictions = new double[nodeIds.size()];
        
        // Get feature schema from DataFrame
        StructType schema = structuralDf.schema();
        
        // Apply the model to make predictions
        for (int i = 0; i < nodeIds.size(); i++) {
            double[] rowData = new double[structuralDf.ncol()];
            for (int j = 0; j < structuralDf.ncol(); j++) {
                rowData[j] = structuralDf.getDouble(i, j);
            }
            
            Tuple inputTuple = Tuple.of(schema, rowData);
            
            if (bestModel instanceof GradientTreeBoost) {
                predictions[i] = ((GradientTreeBoost) bestModel).predict(inputTuple);
            } else if (bestModel instanceof RandomForest) {
                predictions[i] = ((RandomForest) bestModel).predict(inputTuple);
            } else if (bestModel instanceof AdaBoost) {
                predictions[i] = ((AdaBoost) bestModel).predict(inputTuple);
            } else if (bestModel instanceof DecisionTree) {
                predictions[i] = ((DecisionTree) bestModel).predict(inputTuple);
            } else if (bestModel instanceof LinearModel) {
                predictions[i] = ((LinearModel) bestModel).predict(rowData);
            } else {
                throw new IllegalArgumentException("Unsupported model type");
            }
        }
        
        // Create node-score pairs
        List<Map.Entry<String, Double>> scoredNodes = new ArrayList<>();
        for (int i = 0; i < nodeIds.size(); i++) {
            scoredNodes.add(Map.entry(nodeIds.get(i), predictions[i]));
        }
        
        // Sort by score (highest first)
        scoredNodes.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        
        // Extract top-k nodes (use effectiveK instead of k)
        return scoredNodes.stream()
                .limit(effectiveK)
                .map(Map.Entry::getKey)
                .toList();
    }
    
    // Calculate a heuristic importance score based on centrality measures
    private double[] computeHeuristicScore(Map<String, Map<String, Double>> centralityScores, List<String> nodeIds) {
        double[] scores = new double[nodeIds.size()];
        for (int i = 0; i < nodeIds.size(); i++) {
            final int index = i;
            scores[i] = centralityScores.values().stream()
                    .mapToDouble(m -> m.getOrDefault(nodeIds.get(index), 0.0))
                    .sum();
        }
        return scores;
    }
    
    // Combined structural and semantic importance score
    private double[] computeHybridImportanceScore(
            Map<String, Map<String, Double>> centralityScores, 
            List<String> nodeIds) {
        
        double[] scores = new double[nodeIds.size()];
        
        for (int i = 0; i < nodeIds.size(); i++) {
            String nodeId = nodeIds.get(i);
            
            // Structural component - sum of structural centrality measures
            double structuralScore = 0.0;
            int structuralCount = 0;
            
            for (Map.Entry<String, Map<String, Double>> entry : centralityScores.entrySet()) {
                // Skip semantic measures
                if (entry.getKey().equals("semanticPageRank") || 
                    entry.getKey().equals("embeddingDiversity")) {
                    continue;
                }
                
                structuralScore += entry.getValue().getOrDefault(nodeId, 0.0);
                structuralCount++;
            }
            
            if (structuralCount > 0) {
                structuralScore /= structuralCount; // Normalize
            }
            
            // Semantic component - based on embedding properties
            double semanticScore = 0.0;
            
            // Use semantic centrality measures if available
            if (centralityScores.containsKey("semanticPageRank")) {
                semanticScore += centralityScores.get("semanticPageRank").getOrDefault(nodeId, 0.0);
            }
            
            if (centralityScores.containsKey("embeddingDiversity")) {
                semanticScore += centralityScores.get("embeddingDiversity").getOrDefault(nodeId, 0.0);
            }
            
            // Additional semantic score from embedding similarity when available
            if (embeddings != null && embeddings.containsKey(nodeId)) {
                semanticScore += calculateSemanticImportance(nodeId);
            }
            
            // Combined score with weighted alpha
            scores[i] = alpha * structuralScore + (1 - alpha) * semanticScore;
        }
        
        return scores;
    }
    
    // Calculate semantic importance of a node based on its embedding
    private double calculateSemanticImportance(String nodeId) {
        if (embeddings == null || !embeddings.containsKey(nodeId)) {
            return 0.0;
        }
        
        double[] nodeEmbedding = embeddings.get(nodeId);
        double totalSimilarity = 0.0;
        int count = 0;
        
        // Calculate average similarity to other nodes
        for (Map.Entry<String, double[]> entry : embeddings.entrySet()) {
            if (!entry.getKey().equals(nodeId)) {
                double similarity = cosineSimilarity(nodeEmbedding, entry.getValue());
                totalSimilarity += similarity;
                count++;
            }
        }
        
        // Average similarity
        double avgSimilarity = (count > 0) ? totalSimilarity / count : 0.0;
        
        // Diversity score (1 - avgSimilarity) gives higher values to more unique nodes
        return 1.0 - avgSimilarity;
    }
    
    // Utility method to calculate cosine similarity
    private double cosineSimilarity(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }
        
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
    
    // Convert a 1D array to a 2D array (for DataFrame creation)
    private double[][] transformTo2D(double[] original) {
        double[][] transformed = new double[original.length][1];
        for (int i = 0; i < original.length; i++) {
            transformed[i][0] = original[i];
        }
        return transformed;
    }
    
    // Train and select the best regression model
    private ModelWrapper trainBestRegressor(DataFrame df, String targetColumn) 
            throws InstantiationException, IllegalAccessException, IllegalArgumentException, 
                   InvocationTargetException, SecurityException {
        
        System.out.println("Training regression models for node importance prediction...");
        
        DataFrame dfFeatures = df.drop(targetColumn);
        double[] y = df.column(targetColumn).toDoubleArray();
        Formula formula = Formula.lhs(targetColumn);
        
        // Create regression models
        Map<String, ModelWrapper> models = new HashMap<>();
        
        try {
            models.put("Gradient Boosting", new ModelWrapper("GradientTreeBoost", 
                      GradientTreeBoost.fit(formula, df)));
        } catch (Exception e) {
            System.err.println("Error creating Gradient Boosting model: " + e.getMessage());
        }
        
        try {
            models.put("Random Forest", new ModelWrapper("RandomForest", 
                      RandomForest.fit(formula, df)));
        } catch (Exception e) {
            System.err.println("Error creating Random Forest model: " + e.getMessage());
        }
        
        try {
            models.put("Linear OLS", new ModelWrapper("OLS", 
                      OLS.fit(formula, df)));
        } catch (Exception e) {
            System.err.println("Error creating OLS model: " + e.getMessage());
        }
        
        try {
            models.put("Bayesian Ridge", new ModelWrapper("RidgeRegression", 
                      RidgeRegression.fit(formula, df, 0.1)));
        } catch (Exception e) {
            System.err.println("Error creating Ridge Regression model: " + e.getMessage());
        }
        
        try {
            models.put("ElasticNet", new ModelWrapper("ElasticNet", 
                      ElasticNet.fit(formula, df, 1.0, 0.5)));
        } catch (Exception e) {
            System.err.println("Error creating ElasticNet model: " + e.getMessage());
        }
        
        if (models.isEmpty()) {
            throw new RuntimeException("No regression models could be created");
        }
        
        System.out.println("Models: " + models);
        
        // Cross-validate and find best model
        double bestScore = Double.NEGATIVE_INFINITY;
        String bestModelName = "";
        ModelWrapper bestModel = null;
        
        System.out.println("Evaluating models with cross-validation...");
        
        for (Map.Entry<String, ModelWrapper> entry : models.entrySet()) {
            String modelName = entry.getKey();
            ModelWrapper modelWrapper = (ModelWrapper) entry.getValue();
            
            try {
                double score = crossValidate(modelWrapper, dfFeatures.toArray(), y);
                System.out.println("  " + modelName + " score: " + score);
                
                if (score > bestScore) {
                    bestScore = score;
                    bestModelName = modelName;
                    bestModel = modelWrapper;
                }
            } catch (Exception e) {
                System.err.println("Error cross-validating " + modelName + ": " + e.getMessage());
            }
        }
        
        System.out.println("Best model: " + bestModelName + " with score: " + bestScore);
        return bestModel;
    }
    
    // Perform cross-validation for model evaluation
    private double crossValidate(ModelWrapper model, double[][] x, double[] y) 
            throws InstantiationException, IllegalAccessException, InvocationTargetException {
        
        int folds = 5;
        int total = x.length;
        int foldSize = total / folds;
        double mse = 0.0;
        
        for (int i = 0; i < folds; i++) {
            // Split data into training and testing sets
            List<Integer> trainIndices = new ArrayList<>();
            List<Integer> testIndices = new ArrayList<>();
            
            for (int j = 0; j < total; j++) {
                if (j >= i * foldSize && j < (i + 1) * foldSize) {
                    testIndices.add(j);
                } else {
                    trainIndices.add(j);
                }
            }
            
            // Convert indices to arrays
            double[][] xTrain = new double[trainIndices.size()][];
            double[] yTrain = new double[trainIndices.size()];
            double[][] xTest = new double[testIndices.size()][];
            double[] yTest = new double[testIndices.size()];
            
            for (int j = 0; j < trainIndices.size(); j++) {
                int idx = trainIndices.get(j);
                xTrain[j] = x[idx];
                yTrain[j] = y[idx];
            }
            
            for (int j = 0; j < testIndices.size(); j++) {
                int idx = testIndices.get(j);
                xTest[j] = x[idx];
                yTest[j] = y[idx];
            }
            
            // Create training DataFrame
            String[] xColumns = new String[xTrain[0].length];
            for (int j = 0; j < xColumns.length; j++) {
                xColumns[j] = "feature_" + j;
            }
            
            DataFrame trainDf = DataFrame.of(xTrain, xColumns);
            DataFrame yTrainDf = DataFrame.of(transformTo2D(yTrain), new String[]{"Y"});
            DataFrame trainWithY = trainDf.merge(yTrainDf);
            
            // Train model
            ModelWrapper trainedModel = retrainModel(model, trainWithY);
            
            // Evaluate model
            double foldError = 0.0;
            for (int j = 0; j < xTest.length; j++) {
                double prediction = 0.0;
                
                // Create test tuple correctly
                Tuple testTuple = Tuple.of(trainDf.schema(), xTest[j]);
                
                // Make prediction based on model type
                if (trainedModel.model instanceof GradientTreeBoost) {
                    prediction = ((GradientTreeBoost) trainedModel.model).predict(testTuple);
                } else if (trainedModel.model instanceof RandomForest) {
                    prediction = ((RandomForest) trainedModel.model).predict(testTuple);
                } else if (trainedModel.model instanceof AdaBoost) {
                    prediction = ((AdaBoost) trainedModel.model).predict(testTuple);
                } else if (trainedModel.model instanceof DecisionTree) {
                    prediction = ((DecisionTree) trainedModel.model).predict(testTuple);
                } else if (trainedModel.model instanceof LinearModel) {
                    prediction = ((LinearModel) trainedModel.model).predict(xTest[j]);
                }
                
                foldError += Math.pow(prediction - yTest[j], 2);
            }
            
            // Add MSE for this fold
            mse += foldError / xTest.length;
        }
        
        // Return negative MSE (higher is better)
        return -mse / folds;
    }
    
    // Retrain a model on a subset of data
    private ModelWrapper retrainModel(ModelWrapper modelWrapper, DataFrame df) 
            throws InstantiationException, IllegalAccessException, InvocationTargetException {
        
        String modelId = modelWrapper.identifier;
        Formula formula = Formula.lhs("Y");
        
        // Create a new model based on the type
        if ("GradientTreeBoost".equals(modelId)) {
            return new ModelWrapper("GradientTreeBoost", GradientTreeBoost.fit(formula, df));
        } else if ("RandomForest".equals(modelId)) {
            return new ModelWrapper("RandomForest", RandomForest.fit(formula, df));
        } else if ("OLS".equals(modelId)) {
            return new ModelWrapper("OLS", OLS.fit(formula, df));
        } else if ("RidgeRegression".equals(modelId)) {
            return new ModelWrapper("RidgeRegression", RidgeRegression.fit(formula, df, 0.1));
        } else if ("ElasticNet".equals(modelId)) {
            return new ModelWrapper("ElasticNet", ElasticNet.fit(formula, df, 1.0, 0.5));
        } else {
            throw new IllegalArgumentException("Unsupported model type: " + modelId);
        }
    }
}
