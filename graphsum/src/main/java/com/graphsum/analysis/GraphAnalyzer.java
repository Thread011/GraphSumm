package com.graphsum.analysis;

import org.jgrapht.Graph;
import org.jgrapht.alg.scoring.BetweennessCentrality;
import org.jgrapht.alg.scoring.ClosenessCentrality;
//import org.jgrapht.alg.scoring.ClosenessCentrality;
import org.jgrapht.alg.scoring.HarmonicCentrality;
import org.jgrapht.alg.scoring.EigenvectorCentrality;
import org.jgrapht.alg.scoring.PageRank;
import org.jgrapht.alg.connectivity.ConnectivityInspector;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleDirectedWeightedGraph;
import org.jgrapht.graph.DefaultWeightedEdge;
import com.graphsum.visualization.GraphVisualizer;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class GraphAnalyzer {
    private final Graph<String, DefaultEdge> graph;
    private final Map<String, String> nodeLabels;
    private Map<String, double[]> embeddings;

    public GraphAnalyzer(String embeddingsPath) {
        this.graph = new DefaultDirectedGraph<>(DefaultEdge.class);
        this.nodeLabels = new HashMap<>();
        if (embeddingsPath != null) {
            this.embeddings = loadEmbeddings(embeddingsPath);
        } else {
            this.embeddings = null;
        }
    }

    public GraphAnalyzer() {
        this(null);
    }

    public void addEntity(String entityId, String label) {
        if (!graph.containsVertex(entityId)) {
            graph.addVertex(entityId);
            nodeLabels.put(entityId, label);
        }
    }

    public void addRelationship(String sourceId, String targetId) {
        addEntity(sourceId, sourceId);
        addEntity(targetId, targetId);
        if (!graph.containsEdge(sourceId, targetId)) {
            graph.addEdge(sourceId, targetId);
        }
    }

    public Map<String, Map<String, Double>> calculateCentralityMeasures() {
        Map<String, Map<String, Double>> allMeasures = new HashMap<>();
        
        // Print graph statistics
        System.out.println("\nGraph Statistics:");
        System.out.println("Number of vertices: " + graph.vertexSet().size());
        System.out.println("Number of edges: " + graph.edgeSet().size());
        
        // Check connectivity
        ConnectivityInspector<String, DefaultEdge> inspector = 
            new ConnectivityInspector<>(graph);
        List<Set<String>> components = inspector.connectedSets();
        System.out.println("Number of connected components: " + components.size());
        System.out.println("Is graph connected? " + inspector.isConnected());
        
        // Print component sizes
        System.out.println("\nComponent sizes:");
        components.forEach(component -> 
            System.out.println("Component size: " + component.size()));
        
        // Calculate centrality measures
        Map<String, Double> degreeCentrality = calculateDegreeCentrality();
        Map<String, Double> betweenness = new BetweennessCentrality<>(graph).getScores();
        Map<String, Double> harmonic = new HarmonicCentrality<>(graph).getScores(); //for disconnected graphs instead of closeness see: https://jgrapht.org/javadoc/org.jgrapht.core/org/jgrapht/alg/scoring/ClosenessCentrality.html
        Map<String, Double> pageRank = new PageRank<>(graph).getScores();
        Map<String, Double> eigenvector = calculateEigenvectorCentrality();
        Map<String, Double> hits = calculateHITS();
        Map<String, Double> instances = calculateInstances();
        
        // Add semantic centrality if embeddings available
        if (embeddings != null && !embeddings.isEmpty()) {
            Map<String, Double> semanticPageRank = calculateSemanticPageRank();
            allMeasures.put("semanticPageRank", semanticPageRank);
            
            // Add embedding-based diversity measure
            Map<String, Double> embeddingDiversity = calculateEmbeddingDiversity();
            allMeasures.put("embeddingDiversity", embeddingDiversity);
        }
        
        allMeasures.put("degree", degreeCentrality);
        allMeasures.put("betweenness", betweenness);
        allMeasures.put("harmonic", harmonic);
        allMeasures.put("pagerank", pageRank);
        allMeasures.put("eigenvector", eigenvector);
        allMeasures.put("hits", hits);
        allMeasures.put("instances", instances);
        
        // Create single visualization using PageRank for node importance
        try {
            visualizeWithCentrality(pageRank, "business_network.png");
            System.out.println("\nVisualization has been generated as 'business_network.png'");
        } catch (Exception e) {
            System.err.println("Error creating visualization: " + e.getMessage());
            e.printStackTrace();
        }
        
        return allMeasures;
    }
    
    private Map<String, Double> calculateDegreeCentrality() {
        Map<String, Double> degreeCentrality = new HashMap<>();
        graph.vertexSet().forEach(v -> 
            degreeCentrality.put(v, (double) (graph.inDegreeOf(v) + graph.outDegreeOf(v)))
        );
        
        // Normalize
        double maxDegree = degreeCentrality.values().stream().mapToDouble(d -> d).max().orElse(1.0);
        degreeCentrality.replaceAll((k, v) -> v / maxDegree);
        
        return degreeCentrality;
    }
    
    private Map<String, Double> calculateEigenvectorCentrality() {
        try {
            return new EigenvectorCentrality<>(graph, 100).getScores();
        } catch (Exception e) {
            System.out.println("\nEigenvector centrality calculation failed: " + e.getMessage());
            return new HashMap<>();
        }
    }

    private Map<String, Double> calculateInstances() {
        Map<String, Double> normalizedInstances = new HashMap<>();
    
        // Calculate the number of instances dynamically (e.g., based on the number of outgoing edges)
        Map<String, Integer> dynamicInstanceCounts = new HashMap<>();
        graph.vertexSet().forEach(node -> {
            int instanceCount = graph.outgoingEdgesOf(node).size(); // Example: count outgoing edges
            dynamicInstanceCounts.put(node, instanceCount);
        });
    
        // Normalize the instance counts
        double maxInstances = dynamicInstanceCounts.values().stream().mapToInt(i -> i).max().orElse(1);
        dynamicInstanceCounts.forEach((node, count) -> 
            normalizedInstances.put(node, (double) count / maxInstances)
        );
    
        return normalizedInstances;
    }

    private Map<String, Double> calculateHITS() {
        Map<String, Double> hubScores = new HashMap<>();
        Map<String, Double> authorityScores = new HashMap<>();
        
        // Initialize hub and authority scores
        graph.vertexSet().forEach(v -> {
            hubScores.put(v, 1.0);
            authorityScores.put(v, 1.0);
        });

        // Iteratively update hub and authority scores
        int iterations = 100; // Number of iterations
        for (int i = 0; i < iterations; i++) {
            // Update authority scores
            graph.vertexSet().forEach(v -> {
                double authority = graph.incomingEdgesOf(v).stream()
                    .mapToDouble(e -> hubScores.get(graph.getEdgeSource(e)))
                    .sum();
                authorityScores.put(v, authority);
            });

            // Update hub scores
            graph.vertexSet().forEach(v -> {
                double hub = graph.outgoingEdgesOf(v).stream()
                    .mapToDouble(e -> authorityScores.get(graph.getEdgeTarget(e)))
                    .sum();
                hubScores.put(v, hub);
            });

            // Normalize scores
            double maxAuthority = authorityScores.values().stream().mapToDouble(a -> a).max().orElse(1.0);
            double maxHub = hubScores.values().stream().mapToDouble(h -> h).max().orElse(1.0);
            authorityScores.replaceAll((k, v) -> v / maxAuthority);
            hubScores.replaceAll((k, v) -> v / maxHub);
        }

        // Combine hub and authority scores into a single measure
        Map<String, Double> hitsScores = new HashMap<>();
        graph.vertexSet().forEach(v -> 
            hitsScores.put(v, (hubScores.get(v) + authorityScores.get(v)) / 2)
        );

        return hitsScores;
    }
    
    private void visualizeWithCentrality(Map<String, Double> centralityScores, String filename) throws Exception {
        GraphVisualizer.visualizeGraph(graph, nodeLabels, centralityScores, filename);
    }

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
            System.out.printf("Loaded %d embeddings with dimension %d%n", 
                embeddings.size(), 
                embeddings.isEmpty() ? 0 : embeddings.values().iterator().next().length);
        } catch (IOException e) {
            System.err.println("Error loading embeddings: " + e.getMessage());
        }
        return embeddings;
    }

    // Expose graph and nodeLabels for access by the embedding module
    public Graph<String, DefaultEdge> getGraph() {
        return this.graph;
    }

    public Map<String, String> getNodeLabels() {
        return this.nodeLabels;
    }

    private Map<String, Double> calculateSemanticPageRank() {
        if (embeddings == null || embeddings.isEmpty()) {
            // Fall back to regular PageRank if no embeddings
            return new PageRank<>(graph).getScores();
        }
        
        // Create a weighted graph based on embedding similarities
        SimpleDirectedWeightedGraph<String, DefaultWeightedEdge> weightedGraph = 
            new SimpleDirectedWeightedGraph<>(DefaultWeightedEdge.class);
        
        // Add all vertices
        graph.vertexSet().forEach(weightedGraph::addVertex);
        
        // Add edges with weights based on embedding similarity
        for (DefaultEdge edge : graph.edgeSet()) {
            String source = graph.getEdgeSource(edge);
            String target = graph.getEdgeTarget(edge);
            
            double similarity = 1.0; // Default weight
            
            // Calculate similarity if both nodes have embeddings
            if (embeddings.containsKey(source) && embeddings.containsKey(target)) {
                similarity = cosineSimilarity(embeddings.get(source), embeddings.get(target));
                // Transform similarity to ensure positive weights
                similarity = (similarity + 1.0) / 2.0;
            }
            
            DefaultWeightedEdge weightedEdge = weightedGraph.addEdge(source, target);
            weightedGraph.setEdgeWeight(weightedEdge, similarity);
        }
        
        // Run PageRank on the weighted graph
        return new PageRank<>(weightedGraph).getScores();
    }

    private Map<String, Double> calculateEmbeddingDiversity() {
        Map<String, Double> diversityScores = new HashMap<>();
        
        for (String node : graph.vertexSet()) {
            if (!embeddings.containsKey(node)) {
                diversityScores.put(node, 0.0);
                continue;
            }
            
            double[] nodeEmbedding = embeddings.get(node);
            double avgSimilarity = 0.0;
            int count = 0;
            
            for (String otherNode : graph.vertexSet()) {
                if (!node.equals(otherNode) && embeddings.containsKey(otherNode)) {
                    double sim = cosineSimilarity(nodeEmbedding, embeddings.get(otherNode));
                    avgSimilarity += sim;
                    count++;
                }
            }
            
            // Diversity = 1 - average similarity (higher means more unique)
            double diversity = (count > 0) ? 1 - (avgSimilarity / count) : 0.0;
            diversityScores.put(node, diversity);
        }
        
        return diversityScores;
    }

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
}
