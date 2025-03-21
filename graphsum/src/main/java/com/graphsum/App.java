package com.graphsum;

import com.graphsum.client.MultiDataSourceClient;
import com.graphsum.model.WikidataEntity;
import com.graphsum.analysis.GraphAnalyzer;
import com.graphsum.embedding.SemanticEmbedder;
import com.graphsum.embedding.WalkingStrategy;
import com.graphsum.selection.HybridNodeSelection;
import com.graphsum.visualization.GraphVisualizer;
import com.graphsum.visualization.LoadingSpinner;
import com.graphsum.client.QueryLogGenerator;
import java.io.File;
import java.util.Map;
import java.util.List;
import java.util.Set;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.Arrays;

public class App {
    public static void main(String[] args) {
        try {

            System.out.println("\nWelcome to GraphSum!\n");

            // Parse command line arguments
            WalkingStrategy walkingStrategy = WalkingStrategy.RANDOM_WALKS; // Default
            boolean runNodeSelection = false;
            int k = 5; // Default number of nodes to select
            double percentage = -1; // Default to not using percentage
            boolean usePercentage = false;
            boolean nodesOnly = false; // Default to showing edges in visualization
            boolean generateQueryLogs = false; // Default to not generating query logs
            int numQueries = 1000; // Default number of queries to generate
            double alpha = 0.5; // Default balance between structural and semantic importance
            String dataSource = "business"; // Default data source (Use Case 1)
            
            for (int i = 0; i < args.length; i++) {
                if (args[i].equals("--walking-strategy") && i + 1 < args.length) {
                    try {
                        walkingStrategy = WalkingStrategy.valueOf(args[i + 1].toUpperCase());
                    } catch (IllegalArgumentException e) {
                        System.err.println("Invalid walking strategy: " + args[i + 1]);
                        System.err.println("Available strategies: " + 
                            Arrays.stream(WalkingStrategy.values())
                                .map(WalkingStrategy::name)
                                .collect(Collectors.joining(", ")));
                        return;
                    }
                    i++; // Skip the next argument
                } else if (args[i].equals("--select-nodes")) {
                    runNodeSelection = true;
                } else if (args[i].equals("--k") && i + 1 < args.length) {
                    k = Integer.parseInt(args[i + 1]);
                    usePercentage = false;
                    i++;
                } else if (args[i].equals("--percentage") && i + 1 < args.length) {
                    percentage = Double.parseDouble(args[i + 1]);
                    if (percentage <= 0 || percentage > 100) {
                        System.err.println("Percentage must be between 0 and 100");
                        return;
                    }
                    usePercentage = true;
                    nodesOnly = true; // By default, percentage-based selection shows only nodes
                    i++;
                } else if (args[i].equals("--nodes-only")) {
                    nodesOnly = true;
                } else if (args[i].equals("--generate-query-logs")) {
                    generateQueryLogs = true;
                } else if (args[i].equals("--num-queries") && i + 1 < args.length) {
                    numQueries = Integer.parseInt(args[i + 1]);
                    i++;
                } else if (args[i].equals("--alpha") && i + 1 < args.length) {
                    alpha = Double.parseDouble(args[i + 1]);
                    if (alpha < 0 || alpha > 1) {
                        System.err.println("Alpha must be between 0 and 1");
                        return;
                    }
                    i++;
                } else if (args[i].equals("--data-source") && i + 1 < args.length) {
                    dataSource = args[i + 1].toLowerCase();
                    if (!Arrays.asList("business", "academic").contains(dataSource)) {
                        System.err.println("Invalid data source: " + args[i + 1]);
                        System.err.println("Available data sources: business, academic");
                        return;
                    }
                    i++;
                }
            }
            
            // Create the client and analyzer
            MultiDataSourceClient multiClient = new MultiDataSourceClient();
            GraphAnalyzer analyzer = new GraphAnalyzer();
            
            // Fetch entities and their relationships based on selected data source
            Map<WikidataEntity, List<String>> entityRelations;
            String datasetDescription;
            
            System.out.println("Using data source: " + dataSource);
            
            try {
                Map<WikidataEntity, List<String>> tempEntityRelations;
                String tempDatasetDescription;
                
                switch (dataSource) {
                    case "academic":
                        System.out.println("Fetching academic research network (medium dataset)...");
                        tempEntityRelations = multiClient.fetchAcademicNetwork();
                        tempDatasetDescription = "academic research network";
                        break;
                    case "business":
                    default:
                        System.out.println("Fetching business entities network (small dataset)...");
                        tempEntityRelations = multiClient.fetchBusinessEntities();
                        tempDatasetDescription = "business entities network";
                        break;
                }
                
                // Assign to the outer variables after all processing
                entityRelations = tempEntityRelations;
                datasetDescription = tempDatasetDescription;
                
            } catch (Exception e) {
                System.err.println("Error processing knowledge graph: " + e.getMessage());
                e.printStackTrace();
                return;
            }
            
            System.out.println("Building graph from " + datasetDescription + "...");
            
            // Build the graph
            entityRelations.forEach((entity, relations) -> {
                String entityId = entity.getId();
                String label = entity.getLabel().getValue();
                analyzer.addEntity(entityId, label);
                System.out.printf("Added entity: %s (%s)%n", label, entityId);
                
                relations.forEach(relatedId -> {
                    analyzer.addEntity(relatedId, 
                        entityRelations.keySet().stream()
                            .filter(e -> e.getId().equals(relatedId))
                            .findFirst()
                            .map(e -> e.getLabel().getValue())
                            .orElse(relatedId));
                    analyzer.addRelationship(entityId, relatedId);
                    System.out.printf("  Added relationship: %s -> %s%n", entityId, relatedId);
                });
            });
            
            System.out.println("\nCalculating centrality measures and generating visualizations...");
            
            // Calculate and output centrality measures
            Map<String, Map<String, Double>> measures = analyzer.calculateCentralityMeasures();
            measures.forEach((measure, scores) -> {
                System.out.printf("%n%s centrality:%n", measure);
                scores.entrySet().stream()
                    .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                    .limit(5)  // Show top 5 entities for each measure
                    .forEach(entry -> {
                        String entityId = entry.getKey();
                        double score = entry.getValue();
                        String label = entityRelations.keySet().stream()
                            .filter(e -> e.getId().equals(entityId))
                            .findFirst()
                            .map(e -> e.getLabel().getValue())
                            .orElse(entityId);
                        System.out.printf("  %s (%s): %.4f%n", label, entityId, score);
                    });
            });
            
            // Generate semantic embeddings using jRDF2Vec
            System.out.println("\nGenerating semantic embeddings using jRDF2Vec with built-in training...");
            System.out.println("This may take a few minutes depending on the size of the graph and the number of entities.");
            
            // Create working directory for jRDF2Vec
            String workingDir = "./jrdf2vec_workspace";
            new File(workingDir).mkdirs();
            
            // Create logs directory
            String logsDir = "./logs";
            new File(logsDir).mkdirs();
            
            // Initialize the semantic embedder with the specified walking strategy
            System.out.println("Using walking strategy: " + walkingStrategy.name());
            System.out.println("This strategy uses " + walkingStrategy.getParamValue() + " walks for generating embeddings.");
            
            // If using WEIGHTED_WALKS, provide additional information
            if (walkingStrategy == WalkingStrategy.WEIGHTED_WALKS) {
                System.out.println("Note: Weighted walks use edge weights to guide the random walks. " +
                                  "Since your graph doesn't have explicit weights, uniform weights will be used.");
            }
            
            SemanticEmbedder embedder = new SemanticEmbedder(
                workingDir,
                100,  // dimensions
                50,   // number of walks
                4,    // walk depth
                walkingStrategy
            );
            
            // Convert WikidataEntity set to a Set for the embedder
            Set<WikidataEntity> entitySet = entityRelations.keySet();
            
            // Declare embeddingsFile variable outside the try block so it's in scope for node selection
            String embeddingsFile = null;
            
            // Generate embeddings
            try {
                System.out.println("Evaluating different walking strategies to find the optimal one...");
                // Evaluate different walking strategies (optional - can be time-consuming)
                // WalkingStrategy bestStrategy = embedder.evaluateWalkingStrategies(entitySet, analyzer.getGraph(), analyzer.getNodeLabels());
                // System.out.println("Selected optimal walking strategy: " + bestStrategy.name());

                long startTime = System.currentTimeMillis();
                LoadingSpinner spinner = new LoadingSpinner();
                Thread spinnerThread = new Thread(spinner);
                spinnerThread.start();
                
                embeddingsFile = embedder.generateEmbeddings(
                    entitySet, 
                    analyzer.getGraph(), 
                    analyzer.getNodeLabels()
                );
                
                
                // Stop the spinner when the embeddings generation ends
                spinner.stop();
                try {
                    spinnerThread.join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                //Stop the timer and print the time taken
                long elapsedTime = (System.currentTimeMillis() - startTime) / 1000;

                System.out.println("\nEmbeddings generated successfully at: " + embeddingsFile + ". Time elapsed: " + elapsedTime + " seconds");
                
                // Load the generated embeddings
                Map<String, double[]> embeddings = embedder.loadEmbeddings(embeddingsFile);
                
                if (embeddings.isEmpty()) {
                    System.out.println("Warning: No embeddings were generated.");
                } else {
                    System.out.printf("\nGenerated embeddings for %d entities with %d dimensions%n", 
                        embeddings.size(), embeddings.values().iterator().next().length);
                    
                    // Find similar entities for a sample entity
                    if (!entitySet.isEmpty()) {
                        // Get the first entity URI
                        WikidataEntity sampleEntity = entitySet.iterator().next();
                        String sampleEntityUri = sampleEntity.getUri();
                        String sampleEntityLabel = sampleEntity.getLabel().getValue();
                        
                        try {
                            // Check if the entity exists in embeddings
                            if (!embeddings.containsKey(sampleEntityUri)) {
                                System.out.printf("Warning: Entity %s not found in embeddings.%n", sampleEntityUri);
                                
                                // If no entities match, pick the first available one
                                if (!embeddings.isEmpty()) {
                                    sampleEntityUri = embeddings.keySet().iterator().next();
                                    System.out.printf("Using alternative entity for similarity: %s%n", sampleEntityUri);
                                } else {
                                    throw new IllegalStateException("No embeddings available for similarity calculation");
                                }
                            }
                            
                            List<Map.Entry<String, Double>> similarEntities = 
                                embedder.findSimilarEntities(sampleEntityUri, embeddings, 5);
                            
                            System.out.printf("\nTop 5 similar entities to %s (%s):%n", sampleEntityLabel, sampleEntityUri);
                            similarEntities.forEach(entry -> {
                                // Try to find the label for this entity
                                String entityUri = entry.getKey();
                                String label = entityRelations.keySet().stream()
                                    .filter(e -> e.getUri().equals(entityUri))
                                    .findFirst()
                                    .map(e -> e.getLabel().getValue())
                                    .orElse(entityUri);
                                
                                System.out.printf("  %s (%s): %.4f%n", label, entityUri, entry.getValue());
                            });
                        } catch (Exception e) {
                            System.err.println("Error finding similar entities: " + e.getMessage());
                            e.printStackTrace();
                        }
                    }
                }
            } catch (Exception e) {
                System.err.println("Error generating embeddings: " + e.getMessage());
                e.printStackTrace();
            }
            
            // Generate query logs if requested
            String logFile = null;
            QueryLogGenerator queryLogGenerator = null;
            final Map<String, Integer> nodeAccessCount = new HashMap<>();
            
            if (generateQueryLogs) {
                System.out.println("\n-------------------------------");
                System.out.println("Generating Query Logs");
                System.out.println("-------------------------------");
                
                try {
                    // Create query log generator with alpha parameter removed
                    queryLogGenerator = new QueryLogGenerator(logsDir);
                    
                    // Set embeddings if available, but we won't use them for query generation
                    if (embeddingsFile != null) {
                        Map<String, double[]> embeddings = embedder.loadEmbeddings(embeddingsFile);
                        queryLogGenerator.setEmbeddings(embeddings);
                    }
                    
                    // Generate synthetic query logs without using alpha
                    System.out.println("Generating " + numQueries + " synthetic query logs as independent data sources...");
                    System.out.println("These logs will be used later for ML regression model training.");
                    logFile = queryLogGenerator.generateSyntheticQueryLogs(
                        analyzer.getGraph(), 
                        analyzer.getNodeLabels(), 
                        numQueries
                    );
                    
                    System.out.println("Generated query logs: " + logFile);
                    
                    // Get node access counts for later display
                    nodeAccessCount.putAll(queryLogGenerator.getNodeAccessCount());
                    
                } catch (Exception e) {
                    System.err.println("Error generating query logs: " + e.getMessage());
                    e.printStackTrace();
                }
            }
            
            // Perform node selection if requested
            if (runNodeSelection) {
                System.out.println("\n-------------------------------");
                System.out.println("Performing Node Selection");
                System.out.println("-------------------------------");
                
                try {
                    // Determine if we should use embeddings for node selection
                    boolean useEmbeddings = embeddingsFile != null;
                    
                    // Create node selector based on selection method (percentage or k)
                    HybridNodeSelection nodeSelector;
                    if (usePercentage) {
                        nodeSelector = new HybridNodeSelection(percentage, useEmbeddings, embeddingsFile);
                        System.out.println("Using percentage-based selection: " + percentage + "%");
                    } else {
                        nodeSelector = new HybridNodeSelection(k, useEmbeddings, embeddingsFile);
                        System.out.println("Using fixed-size selection: top " + k + " nodes");
                    }
                    
                    // Connect query logs to node selector if available
                    if (queryLogGenerator != null) {
                        System.out.println("Connecting query log data to node selector for ML training...");
                        final QueryLogGenerator finalQueryLogGenerator = queryLogGenerator;
                        finalQueryLogGenerator.connectToNodeSelector(nodeSelector);
                    }
                    
                    // Run selection algorithm
                    List<String> topNodes = nodeSelector.selectTopKNodes(measures);
                    
                    // Display results
                    System.out.println("\nSelected nodes:");
                    for (String nodeId : topNodes) {
                        String label = entityRelations.keySet().stream()
                            .filter(e -> e.getId().equals(nodeId))
                            .findFirst()
                            .map(e -> e.getLabel().getValue())
                            .orElse(nodeId);
                        
                        // Find which centrality measures rank this node highly
                        List<String> topMeasures = new ArrayList<>();
                        for (Map.Entry<String, Map<String, Double>> entry : measures.entrySet()) {
                            Map<String, Double> measureScores = entry.getValue();
                            
                            // Check if this node is in the top 5 for this measure
                            boolean inTop5 = measureScores.entrySet().stream()
                                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                                .limit(5)
                                .anyMatch(e -> e.getKey().equals(nodeId));
                                
                            if (inTop5) {
                                topMeasures.add(entry.getKey());
                            }
                        }
                        
                        // Display node with query count if available
                        if (nodeAccessCount.containsKey(nodeId)) {
                            System.out.printf("  %s (%s) - Query count: %d - Top measures: %s%n", 
                                label, nodeId, nodeAccessCount.getOrDefault(nodeId, 0), 
                                String.join(", ", topMeasures));
                        } else {
                            System.out.printf("  %s (%s) - Top measures: %s%n", 
                                label, nodeId, String.join(", ", topMeasures));
                        }
                    }
                    
                    // Create a map of importance scores for each node based on centrality measures and query logs
                    Map<String, Double> importanceScores = new HashMap<>();
                    
                    // For each node, calculate a combined importance score that includes query frequency
                    for (String nodeId : analyzer.getGraph().vertexSet()) {
                        double combinedScore = 0.0;
                        int measureCount = 0;
                        
                        // Sum up normalized centrality scores
                        for (Map<String, Double> measureScores : measures.values()) {
                            if (measureScores.containsKey(nodeId)) {
                                combinedScore += measureScores.get(nodeId);
                                measureCount++;
                            }
                        }
                        
                        // Calculate the average score if measures were found
                        if (measureCount > 0) {
                            combinedScore /= measureCount;
                        }
                        
                        // Add query log weight if available (higher query frequency = higher importance)
                        if (nodeAccessCount.containsKey(nodeId)) {
                            int queryCount = nodeAccessCount.get(nodeId);
                            // Scale query counts to be in similar range as centrality scores
                            double queryScore = Math.log1p(queryCount); // log(1+x) to handle zero case
                            
                            // Blend scores: 70% centrality-based, 30% query-based
                            combinedScore = (0.7 * combinedScore) + (0.3 * queryScore);
                        }
                        
                        importanceScores.put(nodeId, combinedScore);
                    }
                    
                    // Create visualization based on nodes-only flag
                    try {
                        if (nodesOnly) {
                            System.out.println("\nGenerating visualization of important nodes (without edges)...");
                            
                            // Generate the visualization of important nodes without edges
                            GraphVisualizer.visualizeImportantNodes(
                                analyzer.getNodeLabels(),
                                topNodes,
                                importanceScores,
                                "important_nodes.png"
                            );
                            
                            System.out.println("Visualization of important nodes generated: important_nodes.png");
                        } else {
                            System.out.println("\nGenerating visualization with selected nodes highlighted...");
                            
                            // Create custom scores where selected nodes have value 1.0, others 0.2
                            Map<String, Double> highlightScores = new HashMap<>();
                            for (String nodeId : analyzer.getGraph().vertexSet()) {
                                highlightScores.put(nodeId, topNodes.contains(nodeId) ? 1.0 : 0.2);
                            }
                            
                            // Generate the visualization
                            GraphVisualizer.visualizeGraph(
                                analyzer.getGraph(), 
                                analyzer.getNodeLabels(),
                                highlightScores, 
                                "selected_nodes.png"
                            );
                            
                            System.out.println("Visualization with selected nodes generated: selected_nodes.png");
                        }
                    } catch (Exception e) {
                        System.err.println("Error creating visualization: " + e.getMessage());
                    }
                    
                    
                } catch (Exception e) {
                    System.err.println("Error in node selection: " + e.getMessage());
                    e.printStackTrace();
                }
            }
            
            System.out.println("\nVisualization files have been generated in the current directory.");
            
            // List all generated files
            List<String> generatedFiles = new ArrayList<>();
            generatedFiles.add("business_network.png");
            
            if (runNodeSelection) {
                if (nodesOnly) {
                    generatedFiles.add("important_nodes.png");
                } else {
                    generatedFiles.add("selected_nodes.png");
                }
            }
            
            if (logFile != null) {
                generatedFiles.add(logFile);
            }
            
            System.out.println("Files generated: " + String.join(", ", generatedFiles));
            
            // Only print this if embeddingsFile is not null
            if (embeddingsFile != null) {
                System.out.println("Embeddings have been generated in: " + embeddingsFile);
            }
            
            // Update usage instructions to include all options
            System.out.println("\nUsage options:");
            System.out.println("  --walking-strategy <STRATEGY>   : Set the walking strategy for generating embeddings");
            System.out.println("  --select-nodes                 : Enable node selection");
            System.out.println("  --k <NUMBER>                   : Select top-k nodes by importance (fixed number)");
            System.out.println("  --percentage <NUMBER>          : Select percentage of most important nodes (1-100)");
            System.out.println("  --nodes-only                   : Show only selected nodes without edges");
            System.out.println("  --generate-query-logs          : Generate synthetic query logs for improved node selection");
            System.out.println("  --num-queries <NUMBER>         : Number of synthetic queries to generate (default: 1000)");
            System.out.println("  --alpha <NUMBER>               : Weight between structural (0) and semantic (1) importance (default: 0.5)");
            System.out.println("  --data-source <SOURCE>         : Set the data source (business, academic)");
            
            System.out.println("\nExample commands:");
            System.out.println("For fixed number of nodes:");
            System.out.println("  mvn exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args=\"--walking-strategy RANDOM_WALKS --select-nodes --k 5 --data-source business\"");
            System.out.println("For percentage-based selection with query logs:");
            System.out.println("  mvn exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args=\"--walking-strategy RANDOM_WALKS --select-nodes --percentage 10 --generate-query-logs --num-queries 2000 --data-source academic\"");
            System.out.println("For percentage-based selection with full graph and query logs:");
            System.out.println("  mvn exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args=\"--walking-strategy RANDOM_WALKS --select-nodes --percentage 10 --nodes-only false --generate-query-logs --alpha 0.7 --data-source business\"");
            
        } catch (Exception e) {
            System.err.println("Error processing knowledge graph: " + e.getMessage());
            e.printStackTrace();
        }
    }
}