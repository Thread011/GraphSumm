// package com.graphsum.demo;

// import com.graphsum.analysis.GraphAnalyzer;
// import com.graphsum.client.QueryLogGenerator;
// import com.graphsum.client.WikidataClient;
// import com.graphsum.embedding.SemanticEmbedder;
// import com.graphsum.model.WikidataEntity;
// import com.graphsum.selection.HybridNodeSelection;
// import org.jgrapht.Graph;
// import org.jgrapht.graph.DefaultEdge;

// import java.io.File;
// import java.util.List;
// import java.util.Map;
// import java.util.Set;

// /**
//  * Demo class that shows the integration between QueryLogGenerator and HybridNodeSelection.
//  * It demonstrates how query logs can be used to train ML models for node importance assessment.
//  */
// public class QueryLogDemo {
    
//     public static void main(String[] args) {
//         try {
//             System.out.println("Starting Query Log Demo...");
//             System.out.println("This demo shows how query logs can be used to train ML models for node importance assessment.");
            
//             // Step 1: Create working directories
//             String logsDir = "./query_logs";
//             String embeddingsDir = "./jrdf2vec_workspace";
//             new File(logsDir).mkdirs();
//             new File(embeddingsDir).mkdirs();
            
//             // Step 2: Fetch data and build graph
//             System.out.println("\nFetching Wikidata entities and building graph...");
//             WikidataClient client = new WikidataClient();
//             GraphAnalyzer analyzer = new GraphAnalyzer();
            
//             Map<WikidataEntity, List<String>> entityRelations = client.fetchEntitiesWithRelations();
//             System.out.println("Retrieved " + entityRelations.size() + " entities from Wikidata");
            
//             // Add entities and relationships to the graph
//             entityRelations.forEach((entity, relations) -> {
//                 String entityId = entity.getId();
//                 String label = entity.getLabel().getValue();
//                 analyzer.addEntity(entityId, label);
                
//                 relations.forEach(relatedId -> {
//                     // Find related entity label if possible
//                     String relatedLabel = entityRelations.keySet().stream()
//                         .filter(e -> e.getId().equals(relatedId))
//                         .findFirst()
//                         .map(e -> e.getLabel().getValue())
//                         .orElse(relatedId);
                    
//                     analyzer.addEntity(relatedId, relatedLabel);
//                     analyzer.addRelationship(entityId, relatedId);
//                 });
//             });
            
//             // Get graph information
//             Graph<String, DefaultEdge> graph = analyzer.getGraph();
//             Map<String, String> nodeLabels = analyzer.getNodeLabels();
            
//             System.out.println("Built graph with " + graph.vertexSet().size() + " nodes and " 
//                 + graph.edgeSet().size() + " edges");
            
//             // Step 3: Calculate centrality measures
//             System.out.println("\nCalculating centrality measures...");
//             Map<String, Map<String, Double>> centralityMeasures = analyzer.calculateCentralityMeasures();
            
//             // Step 4: Generate embeddings
//             System.out.println("\nGenerating semantic embeddings...");
//             SemanticEmbedder embedder = new SemanticEmbedder(embeddingsDir, 100, 10, 4); // dimensions=100, walks=10, depth=4
            
//             // Convert entities to a set for embeddings generation
//             Set<WikidataEntity> entitySet = entityRelations.keySet();
            
//             String embeddingsFile = embedder.generateEmbeddings(entitySet, graph, nodeLabels);
//             Map<String, double[]> embeddings = embedder.loadEmbeddings(embeddingsFile);
            
//             System.out.println("Generated embeddings for " + embeddings.size() + " entities");
            
//             // Step 5: Generate query logs
//             System.out.println("\nGenerating synthetic query logs...");
//             double alpha = 0.7; // Weight for structural vs semantic importance
//             QueryLogGenerator generator = new QueryLogGenerator(logsDir, alpha);
            
//             // Set embeddings for semantic importance calculation
//             generator.setEmbeddings(embeddings);
            
//             // Generate synthetic logs (1000 queries)
//             int numQueries = 1000;
//             String logFile = generator.generateSyntheticQueryLogs(graph, nodeLabels, centralityMeasures, numQueries);
//             System.out.println("Generated query logs: " + logFile);
            
//             // Generate feature file for ML training
//             String featureFile = generator.generateFeatureFile(graph, nodeLabels, centralityMeasures);
//             System.out.println("Generated feature file: " + featureFile);
            
//             // Step 6: Set up node selection with ML models
//             System.out.println("\nSetting up node selection with ML models...");
//             int k = 10; // Select top-10 nodes
//             HybridNodeSelection selector = new HybridNodeSelection(k, true, embeddingsFile, alpha);
            
//             // Step 7: Connect query log data to node selector
//             System.out.println("\nConnecting query log data to node selector for ML training...");
//             generator.connectToNodeSelector(selector);
            
//             // Step 8: Select top-K nodes using ML models trained on query logs
//             System.out.println("\nSelecting top-" + k + " nodes using ML models trained on query logs...");
//             List<String> topNodes = selector.selectTopKNodes(centralityMeasures);
            
//             // Step 9: Print results and analyze
//             System.out.println("\nTop-" + k + " important nodes based on ML models trained on query logs:");
//             for (int i = 0; i < topNodes.size(); i++) {
//                 String nodeId = topNodes.get(i);
//                 String label = nodeLabels.getOrDefault(nodeId, nodeId);
//                 int queryCount = generator.getNodeAccessCount().getOrDefault(nodeId, 0);
                
//                 System.out.printf("%d. %s (%s) - Query count: %d\n", 
//                     i + 1, label, nodeId, queryCount);
//             }
            
//             // Step 10: Explain the results
//             System.out.println("\nAnalysis:");
//             System.out.println("- ML models were trained on both structural features (centrality) and behavioral features (query logs)");
//             System.out.println("- Nodes with higher query counts are generally ranked higher by the ML models");
//             System.out.println("- However, ML models also consider other factors like graph structure and embeddings");
//             System.out.println("- This hybrid approach combines user interest (from query logs) with structural importance");
            
//             System.out.println("\nDemo completed successfully!");
            
//         } catch (Exception e) {
//             System.err.println("Error in query log demo: " + e.getMessage());
//             e.printStackTrace();
//         }
//     }
// } 