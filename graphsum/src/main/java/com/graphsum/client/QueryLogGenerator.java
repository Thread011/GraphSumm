package com.graphsum.client;

import com.graphsum.model.WikidataEntity;
import com.graphsum.selection.HybridNodeSelection;
import org.apache.commons.io.FileUtils;
import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultEdge;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

/**
 * Component responsible for generating, storing, and analyzing query logs
 * for training machine learning regressors to assess node importance.
 * Integrates with HybridNodeSelection for consistent importance scoring.
 * 
 * The generation of query logs is completely independent of semantic and structural values.
 * These logs will be used later for ML regression model training.
 */
public class QueryLogGenerator {
    private static final Logger LOGGER = LoggerFactory.getLogger(QueryLogGenerator.class);
    private static final DateTimeFormatter TIMESTAMP_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
    
    private final String logDirectory;
    private final Random random;
    
    // Map of node IDs to their access count
    private final Map<String, Integer> nodeAccessCount;
    // Map of edge pairs to their access count
    private final Map<String, Integer> edgeAccessCount;
    // Map of node IDs to their embeddings (used only for ML training, not for query generation)
    private final Map<String, double[]> embeddings;
    
    /**
     * Creates a new QueryLogGenerator with the specified log directory
     *
     * @param logDirectory Directory to store log files
     */
    public QueryLogGenerator(String logDirectory) {
        this.logDirectory = logDirectory;
        this.random = new Random();
        this.nodeAccessCount = new HashMap<>();
        this.edgeAccessCount = new HashMap<>();
        this.embeddings = new HashMap<>();
        
        // Create log directory if it doesn't exist
        File dir = new File(logDirectory);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }
    
    /**
     * Sets the embeddings for nodes (used only for ML training, not for query generation)
     *
     * @param embeddings Map of node IDs to their embedding vectors
     */
    public void setEmbeddings(Map<String, double[]> embeddings) {
        this.embeddings.clear();
        this.embeddings.putAll(embeddings);
    }
    
    /**
     * Generates synthetic query logs for the given graph.
     * The query generation is completely independent of semantic and structural values.
     * These logs will be used later for ML regression model training.
     *
     * @param graph The RDF graph
     * @param nodeLabels Map of node IDs to their labels
     * @param numQueries Number of synthetic queries to generate
     * @return Path to the generated log file
     * @throws IOException If an error occurs while writing the log file
     */
    public String generateSyntheticQueryLogs(
            Graph<String, DefaultEdge> graph,
            Map<String, String> nodeLabels,
            int numQueries) throws IOException {
        
        List<String> logEntries = new ArrayList<>();
        List<String> nodes = new ArrayList<>(graph.vertexSet());
        
        // Calculate node weights based on graph structure (degree)
        Map<String, Double> nodeWeights = calculateNodeWeights(graph);
        
        // Generate synthetic queries
        for (int i = 0; i < numQueries; i++) {
            // Select a random node with probability partially influenced by degree
            String sourceNode = selectWeightedRandomNode(nodes, nodeWeights);
            
            // Increment node access count
            nodeAccessCount.put(sourceNode, nodeAccessCount.getOrDefault(sourceNode, 0) + 1);
            
            // Select a query type
            String queryType = selectRandomQueryType();
            
            // Generate a timestamp
            String timestamp = LocalDateTime.now()
                    .minusSeconds(ThreadLocalRandom.current().nextInt(0, 86400))
                    .format(TIMESTAMP_FORMAT);
            
            // Create log entry based on query type
            String logEntry;
            
            switch (queryType) {
                case "ENTITY_LOOKUP":
                    logEntry = formatEntityLookupLog(timestamp, sourceNode, nodeLabels.get(sourceNode));
                    break;
                    
                case "PROPERTY_QUERY":
                    logEntry = formatPropertyQueryLog(timestamp, sourceNode, nodeLabels.get(sourceNode), 
                            selectRandomProperty());
                    break;
                    
                case "RELATIONSHIP_QUERY":
                    // Find neighbors
                    List<String> neighbors = getNeighbors(graph, sourceNode);
                    if (!neighbors.isEmpty()) {
                        String targetNode = neighbors.get(random.nextInt(neighbors.size()));
                        // Increment edge access count
                        String edgeKey = sourceNode + "->" + targetNode;
                        edgeAccessCount.put(edgeKey, edgeAccessCount.getOrDefault(edgeKey, 0) + 1);
                        
                        logEntry = formatRelationshipQueryLog(timestamp, sourceNode, nodeLabels.get(sourceNode),
                                targetNode, nodeLabels.get(targetNode));
                    } else {
                        // Fallback to entity lookup if no neighbors
                        logEntry = formatEntityLookupLog(timestamp, sourceNode, nodeLabels.get(sourceNode));
                    }
                    break;
                    
                case "SUBGRAPH_QUERY":
                    logEntry = formatSubgraphQueryLog(timestamp, sourceNode, nodeLabels.get(sourceNode), 
                            random.nextInt(3) + 1);
                    break;
                    
                default:
                    logEntry = formatEntityLookupLog(timestamp, sourceNode, nodeLabels.get(sourceNode));
            }
            
            logEntries.add(logEntry);
        }
        
        // Write log entries to file
        String logFileName = "query_log_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")) + ".log";
        File logFile = new File(logDirectory, logFileName);
        FileUtils.writeLines(logFile, StandardCharsets.UTF_8.name(), logEntries);
        
        LOGGER.info("Generated {} synthetic query logs, saved to {}", numQueries, logFile.getAbsolutePath());
        
        return logFile.getAbsolutePath();
    }
    
    /**
     * Calculates node weights based on graph structure.
     * Uses a combination of degree and randomness for more realistic query patterns.
     * 
     * @param graph The graph structure
     * @return Map of node IDs to their weights
     */
    private Map<String, Double> calculateNodeWeights(Graph<String, DefaultEdge> graph) {
        Map<String, Double> weights = new HashMap<>();
        
        for (String nodeId : graph.vertexSet()) {
            // Calculate degree (in + out)
            int degree = graph.degreeOf(nodeId);
            
            // We want some influence of degree but also randomness
            // 40% based on degree, 60% random to ensure coverage of less connected nodes
            double weight = 0.4 * Math.log1p(degree) + 0.6 * random.nextDouble();
            
            weights.put(nodeId, weight);
        }
        
        return weights;
    }
    
    /**
     * Selects a random node with probability proportional to its weight
     *
     * @param nodes The nodes to select from
     * @param weights The weights of each node
     * @return The selected node ID
     */
    private String selectWeightedRandomNode(List<String> nodes, Map<String, Double> weights) {
        // If no weights or all weights are zero, select uniformly
        boolean allZeroWeights = weights.isEmpty() || weights.values().stream().allMatch(w -> w == 0.0);
        
        if (allZeroWeights) {
            return nodes.get(random.nextInt(nodes.size()));
        }
        
        // Calculate total weight
        double totalWeight = weights.values().stream().mapToDouble(Double::doubleValue).sum();
        
        // Select a random value between 0 and totalWeight
        double value = random.nextDouble() * totalWeight;
        
        // Find the node corresponding to the selected value
        double cumulativeWeight = 0.0;
        for (String node : nodes) {
            cumulativeWeight += weights.getOrDefault(node, 0.0);
            if (cumulativeWeight >= value) {
                return node;
            }
        }
        
        // Fallback to uniform selection
        return nodes.get(random.nextInt(nodes.size()));
    }
    
    /**
     * Gets all neighboring nodes for a given node in the graph
     *
     * @param graph The RDF graph
     * @param nodeId The node ID
     * @return List of neighboring node IDs
     */
    private List<String> getNeighbors(Graph<String, DefaultEdge> graph, String nodeId) {
        Set<String> neighbors = new HashSet<>();
        
        for (DefaultEdge edge : graph.edgesOf(nodeId)) {
            String source = graph.getEdgeSource(edge);
            String target = graph.getEdgeTarget(edge);
            
            if (source.equals(nodeId)) {
                neighbors.add(target);
            } else {
                neighbors.add(source);
            }
        }
        
        return new ArrayList<>(neighbors);
    }
    
    /**
     * Selects a random query type with predefined probabilities
     *
     * @return The selected query type
     */
    private String selectRandomQueryType() {
        double rand = random.nextDouble();
        
        if (rand < 0.4) {
            return "ENTITY_LOOKUP";
        } else if (rand < 0.7) {
            return "PROPERTY_QUERY";
        } else if (rand < 0.9) {
            return "RELATIONSHIP_QUERY";
        } else {
            return "SUBGRAPH_QUERY";
        }
    }
    
    /**
     * Selects a random RDF property
     *
     * @return The selected property
     */
    private String selectRandomProperty() {
        String[] properties = {
            "label", "description", "instanceOf", "subclassOf", "partOf", 
            "hasLocation", "occupation", "industry", "headquarters", "founder"
        };
        
        return properties[random.nextInt(properties.length)];
    }
    
    // Log formatting methods
    
    private String formatEntityLookupLog(String timestamp, String entityId, String label) {
        return String.format("%s [INFO] ENTITY_LOOKUP - User requested entity: %s (%s)", 
                timestamp, label, entityId);
    }
    
    private String formatPropertyQueryLog(String timestamp, String entityId, String label, String property) {
        return String.format("%s [INFO] PROPERTY_QUERY - User requested property '%s' of entity: %s (%s)", 
                timestamp, property, label, entityId);
    }
    
    private String formatRelationshipQueryLog(String timestamp, String sourceId, String sourceLabel, 
                                              String targetId, String targetLabel) {
        return String.format("%s [INFO] RELATIONSHIP_QUERY - User requested relationship between %s (%s) and %s (%s)", 
                timestamp, sourceLabel, sourceId, targetLabel, targetId);
    }
    
    private String formatSubgraphQueryLog(String timestamp, String entityId, String label, int depth) {
        return String.format("%s [INFO] SUBGRAPH_QUERY - User requested subgraph of depth %d around entity: %s (%s)", 
                timestamp, depth, label, entityId);
    }
    
    /**
     * Returns the map of node access counts
     *
     * @return Map of node IDs to their access counts
     */
    public Map<String, Integer> getNodeAccessCount() {
        return new HashMap<>(nodeAccessCount);
    }
    
    /**
     * Returns the map of edge access counts
     *
     * @return Map of edge keys to their access counts
     */
    public Map<String, Integer> getEdgeAccessCount() {
        return new HashMap<>(edgeAccessCount);
    }
    
    /**
     * Connects this QueryLogGenerator to a HybridNodeSelection instance,
     * providing query log data for ML training.
     *
     * @param selector The HybridNodeSelection instance to connect to
     * @return The same HybridNodeSelection instance for chaining
     */
    public HybridNodeSelection connectToNodeSelector(HybridNodeSelection selector) {
        // Pass the query log data to the node selector
        selector.loadQueryLogFeatures(this.nodeAccessCount, this.edgeAccessCount);
        
        LOGGER.info("Connected query log data to node selector: {} node access records, {} edge access records",
                nodeAccessCount.size(), edgeAccessCount.size());
        
        return selector;
    }
} 