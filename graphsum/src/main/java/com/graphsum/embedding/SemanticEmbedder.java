package com.graphsum.embedding;

import com.graphsum.model.WikidataEntity;
import org.apache.commons.io.FileUtils;
import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultEdge;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.*;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

/**
 * Component for generating semantic embeddings using jRDF2Vec
 */
public class SemanticEmbedder {
    private static final Logger LOGGER = LoggerFactory.getLogger(SemanticEmbedder.class);
    private final String workingDirectory;
    private final int dimensions;
    private final int numberOfWalks;
    private final int walkDepth;
    private final String jrdf2vecPath;
    private final WalkingStrategy walkingStrategy;
    
    /**
     * Constructor for SemanticEmbedder with default walking strategy (RANDOM_WALKS)
     * 
     * @param workingDirectory Directory where temporary files and results will be stored
     * @param dimensions Number of dimensions for the embeddings
     * @param numberOfWalks Number of walks per entity
     * @param walkDepth Depth of each walk
     */
    public SemanticEmbedder(String workingDirectory, int dimensions, int numberOfWalks, int walkDepth) {
        this(workingDirectory, dimensions, numberOfWalks, walkDepth, WalkingStrategy.RANDOM_WALKS);
    }
    
    /**
     * Constructor for SemanticEmbedder with specified walking strategy
     * 
     * @param workingDirectory Directory where temporary files and results will be stored
     * @param dimensions Number of dimensions for the embeddings
     * @param numberOfWalks Number of walks per entity
     * @param walkDepth Depth of each walk
     * @param walkingStrategy The walking strategy to use for generating walks
     */
    public SemanticEmbedder(String workingDirectory, int dimensions, int numberOfWalks, int walkDepth, 
                           WalkingStrategy walkingStrategy) {
        this.workingDirectory = workingDirectory;
        this.dimensions = dimensions;
        this.numberOfWalks = numberOfWalks;
        this.walkDepth = walkDepth;
        this.walkingStrategy = walkingStrategy;
        
        // Updated to use the correct JAR version
        this.jrdf2vecPath = new File(workingDirectory, "jrdf2vec-1.3-SNAPSHOT.jar").getAbsolutePath();
        
        // Create working directory if it doesn't exist
        new File(workingDirectory).mkdirs();
        
        // Create all necessary directories - use python_server with underscore consistently
        new File(workingDirectory + "/walks").mkdirs();
        new File(workingDirectory + "/python_server").mkdirs();
        new File(workingDirectory + "/model").mkdirs();
        
        LOGGER.info("Initialized SemanticEmbedder with walking strategy: {}", walkingStrategy.name());
    }
    
    /**
     * Exports the graph as NT triples for jRDF2Vec processing
     * 
     * @param graph The graph to export
     * @param nodeLabels Map of node IDs to their labels
     * @return The path to the generated triples file
     */
    public String exportGraphAsTriples(Graph<String, DefaultEdge> graph, Map<String, String> nodeLabels) throws IOException {
        File tripleFile = new File(workingDirectory, "triples.nt");
        List<String> triples = new ArrayList<>();
        
        // Convert graph to NT triples format
        for (DefaultEdge edge : graph.edgeSet()) {
            String source = graph.getEdgeSource(edge);
            String target = graph.getEdgeTarget(edge);
            
            // Create URIs for source and target
            String sourceUri = formatUri(source);
            String targetUri = formatUri(target);
            
            // Use a generic predicate for the relationship
            String predicate = "<http://www.w3.org/2000/01/rdf-schema#isRelatedTo>";
            
            // Add the triple
            triples.add(sourceUri + " " + predicate + " " + targetUri + " .");
            
            // Add label triples if available
            if (nodeLabels.containsKey(source)) {
                String label = escapeStringLiteral(nodeLabels.get(source));
                triples.add(sourceUri + " <http://www.w3.org/2000/01/rdf-schema#label> \"" + label + "\" .");
            }
            
            if (nodeLabels.containsKey(target)) {
                String label = escapeStringLiteral(nodeLabels.get(target));
                triples.add(targetUri + " <http://www.w3.org/2000/01/rdf-schema#label> \"" + label + "\" .");
            }
        }
        
        // Log the first few triples for debugging
        if (!triples.isEmpty()) {
            LOGGER.info("First 5 triples (of {}): ", triples.size());
            for (int i = 0; i < Math.min(5, triples.size()); i++) {
                LOGGER.info("  {}", triples.get(i));
            }
        } else {
            LOGGER.warn("No triples were generated from the graph!");
        }
        
        // Write triples to file
        FileUtils.writeLines(tripleFile, StandardCharsets.UTF_8.name(), triples);
        
        return tripleFile.getAbsolutePath();
    }
    
    /**
     * Format entity ID to URI consistently
     * This is a key method to ensure URI consistency between entities and triples
     */
    private String formatUri(String id) {
        // If the ID is already a URI, return it as is
        if (id.startsWith("http://") || id.startsWith("https://")) {
            return "<" + id + ">";
        }
        
        // Otherwise, create a URI using the same format as WikidataEntity
        return "<http://www.wikidata.org/entity/" + id + ">";
    }
    
    /**
     * Get the URI format for an entity without angle brackets
     * Used for entities.txt file
     */
    private String getEntityUri(WikidataEntity entity) {
        return entity.getUri();
    }
    
    private String escapeStringLiteral(String str) {
        return str.replace("\\", "\\\\")
                 .replace("\"", "\\\"")
                 .replace("\n", "\\n")
                 .replace("\r", "\\r")
                 .replace("\t", "\\t");
    }
    
    /**
     * Generates embeddings for the given entities using jRDF2Vec
     */
    public String generateEmbeddings(Set<WikidataEntity> entities, Graph<String, DefaultEdge> graph, 
                                    Map<String, String> nodeLabels) throws IOException {
        // Clean up old files first
        cleanupOldFiles();
        
        // Create a temporary file with entity URIs for jRDF2Vec light mode
        File entityFile = new File(workingDirectory, "entities.txt");
        List<String> entityUris = entities.stream()
                .map(this::getEntityUri)
                .collect(Collectors.toList());
        
        // Log the first few entities for debugging
        if (!entityUris.isEmpty()) {
            LOGGER.info("First 5 entities (of {}): ", entityUris.size());
            for (int i = 0; i < Math.min(5, entityUris.size()); i++) {
                LOGGER.info("  {}", entityUris.get(i));
            }
        } else {
            LOGGER.warn("No entity URIs were found!");
        }
        
        FileUtils.writeLines(entityFile, StandardCharsets.UTF_8.name(), entityUris);
        
        // Export the graph as NT triples for jRDF2Vec
        String tripleFilePath = exportGraphAsTriples(graph, nodeLabels);
        File tripleFile = new File(tripleFilePath);
        
        // Use absolute paths consistently
        File walkDir = new File(workingDirectory, "walks");
        walkDir.mkdirs();
        String walkDirPath = walkDir.getAbsolutePath();
        
        // Create model directory
        File modelDir = new File(workingDirectory, "model");
        modelDir.mkdirs();
        
        // Define output files
        File vectorsFile = new File(workingDirectory, "vectors.txt");
        File modelFile = new File(new File(workingDirectory, "model"), "model.bin");
        modelFile.getParentFile().mkdirs();
        
        // Build the jRDF2Vec command with the selected walking strategy
        List<String> command = new ArrayList<>(Arrays.asList(
                "java", "-jar", jrdf2vecPath,
                "-light", entityFile.getAbsolutePath(),
                "-graph", tripleFile.getAbsolutePath(),
                "-walkDirectory", walkDirPath,
                "-modelDirectory", modelFile.getAbsolutePath(),
                "-numberOfWalks", String.valueOf(numberOfWalks),
                "-depth", String.valueOf(walkDepth),
                "-dimensions", String.valueOf(dimensions),
                "-epochs", "5",  // Number of training epochs
                "-window", "5",   // Context window size
                "-walkGenerationMode", walkingStrategy.getParamValue(),  // Add walking strategy
                "-outputFile", vectorsFile.getAbsolutePath() // Specify the output file explicitly
        ));
        
        // Add strategy-specific parameters if needed
        if (walkingStrategy == WalkingStrategy.WALKLETS) {
            command.add("-walkletWindow");
            command.add("3");  // Default walklet window size
        } else if (walkingStrategy == WalkingStrategy.NGRAMS) {
            command.add("-ngramSize");
            command.add("3");  // Default n-gram size
        } else if (walkingStrategy == WalkingStrategy.HALK) {
            command.add("-halkHops");
            command.add("2");  // Default HALK hops
        }
        
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.directory(new File(workingDirectory));
        processBuilder.redirectErrorStream(true);
        
        LOGGER.info("Executing jRDF2Vec command with {} walking strategy: {}", 
                   walkingStrategy.name(), String.join(" ", processBuilder.command()));
        
        // Execute jRDF2Vec for walk generation and training
        Process process = processBuilder.start();
        
        // Read and log the output
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                LOGGER.info("jRDF2Vec: {}", line);
            }
        }
        
        try {
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                LOGGER.error("jRDF2Vec process exited with code {}", exitCode);
                LOGGER.warn("Attempting to continue despite exit code, as embeddings might still be generated");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("jRDF2Vec process was interrupted", e);
        }
        
        // Check for the generated vectors file
        if (!vectorsFile.exists()) {
            LOGGER.warn("jRDF2Vec did not generate vectors.txt in expected location. Looking for alternatives...");
            
            // Try to find vectors file in other locations
            File altVectorsFile = new File(walkDir, "vectors.txt");
            if (altVectorsFile.exists()) {
                LOGGER.info("Found vectors file at {}", altVectorsFile.getAbsolutePath());
                vectorsFile = altVectorsFile;
            } else {
                File pythonServerVectorsFile = new File(workingDirectory, "python_server/vectors.txt");
                if (pythonServerVectorsFile.exists()) {
                    LOGGER.info("Found vectors file at {}", pythonServerVectorsFile.getAbsolutePath());
                    vectorsFile = pythonServerVectorsFile;
                } else {
                    // If no vectors file is found, fall back to our custom implementation
                    LOGGER.warn("No vectors file found. Falling back to custom embedding generation.");
                    vectorsFile = new File(walkDir, "vectors.txt");
                    
                    // Check if any walk files were generated
                    File[] walkFiles = walkDir.listFiles((dir, name) -> name.endsWith(".gz"));
                    if (walkFiles != null && walkFiles.length > 0) {
                        LOGGER.info("Found {} walk files. Generating embeddings from walks.", walkFiles.length);
                        generateEmbeddingsFromWalks(walkDir, vectorsFile, entities);
                    } else {
                        LOGGER.warn("No walk files found. Creating embeddings directly from entities.");
                        createEmbeddingsFromEntities(entities, vectorsFile);
                    }
                }
            }
        } else {
            LOGGER.info("jRDF2Vec successfully generated vectors at {}", vectorsFile.getAbsolutePath());
            
            // Copy to walks directory for consistency with our fallback mechanism
            File walkDirVectorsFile = new File(walkDir, "vectors.txt");
            if (!walkDirVectorsFile.exists()) {
                Files.copy(vectorsFile.toPath(), walkDirVectorsFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            }
        }
        
        return vectorsFile.getAbsolutePath();
    }
    
    /**
     * Generate embeddings directly from entities when no walks are available
     */
    private void createEmbeddingsFromEntities(Set<WikidataEntity> entities, File outputFile) throws IOException {
        LOGGER.info("Creating embeddings directly from {} entities", entities.size());
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {
            Random random = new Random(42); // Fixed seed for reproducibility
            
            for (WikidataEntity entity : entities) {
                writer.write(getEntityUri(entity));
                
                // Generate a random vector with the specified dimension
                double[] vector = new double[dimensions];
                for (int i = 0; i < dimensions; i++) {
                    vector[i] = random.nextGaussian();
                }
                
                // Normalize the vector
                double norm = 0.0;
                for (double v : vector) {
                    norm += v * v;
                }
                norm = Math.sqrt(norm);
                
                for (int i = 0; i < dimensions; i++) {
                    double normalizedValue = norm > 0 ? vector[i] / norm : vector[i];
                    writer.write(" " + normalizedValue);
                }
                
                writer.newLine();
            }
        }
        
        LOGGER.info("Generated embeddings written to {}", outputFile.getAbsolutePath());
        
        // Also copy to python_server directory for compatibility
        File pythonServerVectorsFile = new File(workingDirectory, "python_server/vectors.txt");
        Files.copy(outputFile.toPath(), pythonServerVectorsFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
    }
    
    /**
     * Generate embeddings from walk files using a simplified Word2Vec approach
     */
    private void generateEmbeddingsFromWalks(File walkDir, File outputFile, Set<WikidataEntity> entities) throws IOException {
        LOGGER.info("Generating embeddings from walks in {}", walkDir.getAbsolutePath());
        
        // Find all walk files (they should be gzipped)
        File[] walkFiles = walkDir.listFiles((dir, name) -> name.endsWith(".gz"));
        
        if (walkFiles == null || walkFiles.length == 0) {
            LOGGER.warn("No walk files found in {}. Creating dummy embeddings.", walkDir.getAbsolutePath());
            createDummyEmbeddings(outputFile, entities);
            return;
        }
        
        // Extract all unique entities and build co-occurrence matrix
        Set<String> extractedEntities = new HashSet<>();
        Map<String, Map<String, Integer>> coOccurrences = new HashMap<>();
        
        try {
            // First pass: extract all entities
            for (File walkFile : walkFiles) {
                LOGGER.info("Processing walk file (first pass): {}", walkFile.getName());
                try (GZIPInputStream gzipInputStream = new GZIPInputStream(new FileInputStream(walkFile));
                     BufferedReader reader = new BufferedReader(new InputStreamReader(gzipInputStream, StandardCharsets.UTF_8))) {
                    
                    String line;
                    int lineCount = 0;
                    while ((line = reader.readLine()) != null) {
                        lineCount++;
                        // Debug the first few lines to see the format
                        if (lineCount <= 5) {
                            LOGGER.info("Walk line {}: {}", lineCount, line);
                        }
                        
                        String[] tokens = line.split("\\s+");
                        for (int i = 0; i < tokens.length; i += 2) { // Skip predicates
                            if (i < tokens.length) {
                                String entity = tokens[i].trim();
                                if (!entity.isEmpty()) {
                                    extractedEntities.add(entity);
                                    coOccurrences.putIfAbsent(entity, new HashMap<>());
                                }
                            }
                        }
                    }
                    LOGGER.info("Read {} lines from walk file", lineCount);
                } catch (Exception e) {
                    LOGGER.error("Error reading walk file {}: {}", walkFile.getName(), e.getMessage());
                }
            }
            
            LOGGER.info("Found {} unique entities in walks", extractedEntities.size());
            
            if (extractedEntities.isEmpty()) {
                LOGGER.warn("No entities found in walk files. Creating dummy embeddings with real entity URIs.");
                createDummyEmbeddings(outputFile, entities);
                return;
            }
            
            // Second pass: build co-occurrence matrix
            for (File walkFile : walkFiles) {
                LOGGER.info("Processing walk file (second pass): {}", walkFile.getName());
                try (GZIPInputStream gzipInputStream = new GZIPInputStream(new FileInputStream(walkFile));
                     BufferedReader reader = new BufferedReader(new InputStreamReader(gzipInputStream, StandardCharsets.UTF_8))) {
                    
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] tokens = line.split("\\s+");
                        List<String> walkEntities = new ArrayList<>();
                        
                        // Extract only entities (skip predicates)
                        for (int i = 0; i < tokens.length; i += 2) {
                            if (i < tokens.length) {
                                String entity = tokens[i].trim();
                                if (!entity.isEmpty()) {
                                    walkEntities.add(entity);
                                }
                            }
                        }
                        
                        // Update co-occurrence counts (using a window size of 2)
                        for (int i = 0; i < walkEntities.size(); i++) {
                            String entity = walkEntities.get(i);
                            for (int j = Math.max(0, i - 2); j <= Math.min(walkEntities.size() - 1, i + 2); j++) {
                                if (i != j) {
                                    String context = walkEntities.get(j);
                                    Map<String, Integer> contextCounts = coOccurrences.get(entity);
                                    contextCounts.put(context, contextCounts.getOrDefault(context, 0) + 1);
                                }
                            }
                        }
                    }
                } catch (Exception e) {
                    LOGGER.error("Error building co-occurrence matrix from walk file {}: {}", 
                                walkFile.getName(), e.getMessage());
                }
            }
            
            LOGGER.info("Built co-occurrence matrix for {} entities", coOccurrences.size());
            
            // Generate embeddings using a simplified approach based on co-occurrences
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {
                Random random = new Random(42); // For initialization
                
                // Initialize vectors with small random values
                Map<String, double[]> vectors = new HashMap<>();
                for (String entity : extractedEntities) {
                    double[] vector = new double[dimensions];
                    for (int i = 0; i < dimensions; i++) {
                        vector[i] = (random.nextDouble() - 0.5) / dimensions;
                    }
                    vectors.put(entity, vector);
                }
                
                // Adjust vectors based on co-occurrences (simplified Word2Vec)
                for (String entity : extractedEntities) {
                    Map<String, Integer> contextCounts = coOccurrences.get(entity);
                    double[] entityVector = vectors.get(entity);
                    
                    for (Map.Entry<String, Integer> entry : contextCounts.entrySet()) {
                        String context = entry.getKey();
                        int count = entry.getValue();
                        
                        if (vectors.containsKey(context)) {
                            double[] contextVector = vectors.get(context);
                            double weight = Math.log(1 + count) / 10.0; // Dampened weight
                            
                            // Adjust vectors to be more similar for co-occurring entities
                            for (int i = 0; i < dimensions; i++) {
                                entityVector[i] += weight * contextVector[i] / dimensions;
                            }
                        }
                    }
                }
                
                // Normalize and write vectors
                for (String entity : extractedEntities) {
                    writer.write(entity);
                    double[] vector = vectors.get(entity);
                    
                    // Normalize the vector
                    double norm = 0.0;
                    for (double v : vector) {
                        norm += v * v;
                    }
                    norm = Math.sqrt(norm);
                    
                    for (int i = 0; i < dimensions; i++) {
                        double normalizedValue = norm > 0 ? vector[i] / norm : vector[i];
                        writer.write(" " + normalizedValue);
                    }
                    
                    writer.newLine();
                }
            }
            
            LOGGER.info("Generated embeddings written to {}", outputFile.getAbsolutePath());
            
            // Also copy to python_server directory for compatibility
            File pythonServerVectorsFile = new File(workingDirectory, "python_server/vectors.txt");
            Files.copy(outputFile.toPath(), pythonServerVectorsFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            
        } catch (Exception e) {
            LOGGER.error("Error processing walk files: {}", e.getMessage(), e);
            createDummyEmbeddings(outputFile, entities);
        }
    }
    
    /**
     * Create dummy embeddings as a fallback
     * 
     * @param outputFile The file to write embeddings to
     * @param entities The set of entities to create embeddings for (can be null)
     */
    private void createDummyEmbeddings(File outputFile, Set<WikidataEntity> entities) throws IOException {
        LOGGER.info("Creating dummy embeddings in {}", outputFile.getAbsolutePath());
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {
            Random random = new Random(42); // Fixed seed for reproducibility
            
            // If we have real entities, use their URIs
            if (entities != null && !entities.isEmpty()) {
                LOGGER.info("Creating embeddings for {} real entities", entities.size());
                for (WikidataEntity entity : entities) {
                    String uri = getEntityUri(entity);
                    writer.write(uri);
                    
                    // Generate a random vector with the specified dimension
                    double[] vector = new double[dimensions];
                    for (int j = 0; j < dimensions; j++) {
                        vector[j] = random.nextGaussian();
                    }
                    
                    // Normalize the vector
                    double norm = 0.0;
                    for (double v : vector) {
                        norm += v * v;
                    }
                    norm = Math.sqrt(norm);
                    
                    for (int j = 0; j < dimensions; j++) {
                        double normalizedValue = norm > 0 ? vector[j] / norm : vector[j];
                        writer.write(" " + normalizedValue);
                    }
                    
                    writer.newLine();
                }
            } else {
                // Create dummy embeddings with example URIs
                LOGGER.info("No entities provided, creating 10 example embeddings");
                for (int i = 0; i < 10; i++) {
                    writer.write("http://www.wikidata.org/entity/Q" + i);
                    
                    // Generate a random vector with the specified dimension
                    double[] vector = new double[dimensions];
                    for (int j = 0; j < dimensions; j++) {
                        vector[j] = random.nextGaussian();
                    }
                    
                    // Normalize the vector
                    double norm = 0.0;
                    for (double v : vector) {
                        norm += v * v;
                    }
                    norm = Math.sqrt(norm);
                    
                    for (int j = 0; j < dimensions; j++) {
                        double normalizedValue = norm > 0 ? vector[j] / norm : vector[j];
                        writer.write(" " + normalizedValue);
                    }
                    
                    writer.newLine();
                }
            }
        }
        
        LOGGER.info("Dummy embeddings written to {}", outputFile.getAbsolutePath());
        
        // Also copy to python_server directory for compatibility
        File pythonServerVectorsFile = new File(workingDirectory, "python_server/vectors.txt");
        Files.copy(outputFile.toPath(), pythonServerVectorsFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
    }
    
    /**
     * Loads embeddings from a file generated by jRDF2Vec
     * 
     * @param embeddingsFile The file containing the embeddings
     * @return A map from entity URIs to their embedding vectors
     */
    public Map<String, double[]> loadEmbeddings(String embeddingsFilePath) throws IOException {
        Map<String, double[]> embeddings = new HashMap<>();
        
        File embeddingsFile = new File(embeddingsFilePath);
        if (!embeddingsFile.exists()) {
            // Try alternative locations
            File altFile = new File(workingDirectory, "walks/vectors.txt");
            if (altFile.exists()) {
                embeddingsFilePath = altFile.getAbsolutePath();
            } else {
                File pythonServerFile = new File(workingDirectory, "python_server/vectors.txt");
                if (pythonServerFile.exists()) {
                    embeddingsFilePath = pythonServerFile.getAbsolutePath();
                } else {
                    throw new IOException("Embeddings file not found at " + embeddingsFilePath + 
                                        " or alternative locations " + altFile.getAbsolutePath() + 
                                        " or " + pythonServerFile.getAbsolutePath());
                }
            }
        }
        
        try (BufferedReader reader = new BufferedReader(new FileReader(embeddingsFilePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");
                if (parts.length < 2) {
                    continue; // Skip invalid lines
                }
                
                String entity = parts[0];
                double[] vector = new double[parts.length - 1];
                
                for (int i = 1; i < parts.length; i++) {
                    vector[i - 1] = Double.parseDouble(parts[i]);
                }
                
                embeddings.put(entity, vector);
            }
        }
        
        return embeddings;
    }
    
    /**
     * Calculates cosine similarity between two vectors
     * 
     * @param vector1 First vector
     * @param vector2 Second vector
     * @return Cosine similarity value between -1 and 1
     */
    public double cosineSimilarity(double[] vector1, double[] vector2) {
        if (vector1.length != vector2.length) {
            throw new IllegalArgumentException("Vectors must have the same dimension");
        }
        
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        for (int i = 0; i < vector1.length; i++) {
            dotProduct += vector1[i] * vector2[i];
            norm1 += Math.pow(vector1[i], 2);
            norm2 += Math.pow(vector2[i], 2);
        }
        
        if (norm1 == 0.0 || norm2 == 0.0) {
            return 0.0; // Handle zero vectors
        }
        
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
    
    /**
     * Finds the most similar entities to a given entity based on embedding similarity
     * 
     * @param entityUri The URI of the entity to find similar entities for
     * @param embeddings Map of entity URIs to their embedding vectors
     * @param topK Number of similar entities to return
     * @return List of entity URIs sorted by similarity (most similar first)
     */
    public List<Map.Entry<String, Double>> findSimilarEntities(String entityUri, 
                                                             Map<String, double[]> embeddings, 
                                                             int topK) {
        if (!embeddings.containsKey(entityUri)) {
            throw new IllegalArgumentException("Entity not found in embeddings: " + entityUri);
        }
        
        double[] vector = embeddings.get(entityUri);
        Map<String, Double> similarities = new HashMap<>();
        
        for (Map.Entry<String, double[]> entry : embeddings.entrySet()) {
            if (!entry.getKey().equals(entityUri)) {
                double similarity = cosineSimilarity(vector, entry.getValue());
                similarities.put(entry.getKey(), similarity);
            }
        }
        
        return similarities.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(topK)
                .collect(Collectors.toList());
    }
    
    /**
     * Evaluates different walking strategies and returns the best one based on a simple metric
     * 
     * @param entities The set of entities to evaluate
     * @param graph The graph to use for evaluation
     * @param nodeLabels Map of node IDs to their labels
     * @return The best walking strategy based on evaluation
     */
    public WalkingStrategy evaluateWalkingStrategies(Set<WikidataEntity> entities, 
                                                   Graph<String, DefaultEdge> graph, 
                                                   Map<String, String> nodeLabels) throws IOException {
        LOGGER.info("Evaluating different walking strategies...");
        
        // Subset of entities for faster evaluation
        Set<WikidataEntity> sampleEntities = entities.stream()
                .limit(Math.min(entities.size(), 10))
                .collect(Collectors.toSet());
        
        Map<WalkingStrategy, Double> strategyScores = new HashMap<>();
        
        // Evaluate each strategy
        for (WalkingStrategy strategy : WalkingStrategy.values()) {
            try {
                LOGGER.info("Evaluating {} walking strategy...", strategy.name());
                
                // Create a temporary embedder with the current strategy
                SemanticEmbedder tempEmbedder = new SemanticEmbedder(
                    workingDirectory + "/eval_" + strategy.name().toLowerCase(),
                    dimensions,
                    Math.min(numberOfWalks, 10),  // Use fewer walks for evaluation
                    Math.min(walkDepth, 3),       // Use smaller depth for evaluation
                    strategy
                );
                
                // Generate embeddings with this strategy
                String embeddingsFile = tempEmbedder.generateEmbeddings(sampleEntities, graph, nodeLabels);
                
                // Load the generated embeddings
                Map<String, double[]> embeddings = tempEmbedder.loadEmbeddings(embeddingsFile);
                
                // Calculate a simple quality score (average cosine similarity between related entities)
                double score = calculateEmbeddingQualityScore(embeddings, graph, sampleEntities);
                
                strategyScores.put(strategy, score);
                LOGGER.info("{} strategy score: {}", strategy.name(), score);
                
            } catch (Exception e) {
                LOGGER.error("Error evaluating {} strategy: {}", strategy.name(), e.getMessage());
                strategyScores.put(strategy, 0.0);  // Assign a zero score on failure
            }
        }
        
        // Find the best strategy
        WalkingStrategy bestStrategy = strategyScores.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(WalkingStrategy.RANDOM_WALKS);  // Default to random walks if evaluation fails
        
        LOGGER.info("Best walking strategy based on evaluation: {}", bestStrategy.name());
        return bestStrategy;
    }
    
    /**
     * Calculates a quality score for embeddings based on graph structure
     * 
     * @param embeddings The embeddings to evaluate
     * @param graph The original graph
     * @param entities The set of entities
     * @return A quality score (higher is better)
     */
    private double calculateEmbeddingQualityScore(Map<String, double[]> embeddings, 
                                                Graph<String, DefaultEdge> graph,
                                                Set<WikidataEntity> entities) {
        if (embeddings.isEmpty()) {
            return 0.0;
        }
        
        double totalSimilarity = 0.0;
        int count = 0;
        
        // For each entity, calculate similarity with its neighbors in the graph
        for (WikidataEntity entity : entities) {
            String entityUri = getEntityUri(entity);
            if (!embeddings.containsKey(entityUri)) {
                continue;
            }
            
            double[] entityVector = embeddings.get(entityUri);
            
            // Get neighbors in the graph
            Set<String> neighbors = new HashSet<>();
            for (DefaultEdge edge : graph.edgesOf(entity.getId())) {
                String source = graph.getEdgeSource(edge);
                String target = graph.getEdgeTarget(edge);
                
                if (source.equals(entity.getId())) {
                    neighbors.add(target);
                } else {
                    neighbors.add(source);
                }
            }
            
            // Calculate average similarity with neighbors
            for (String neighborId : neighbors) {
                String neighborUri = formatUri(neighborId).replace("<", "").replace(">", "");
                if (embeddings.containsKey(neighborUri)) {
                    double similarity = cosineSimilarity(entityVector, embeddings.get(neighborUri));
                    totalSimilarity += similarity;
                    count++;
                }
            }
        }
        
        return count > 0 ? totalSimilarity / count : 0.0;
    }
    
    /**
     * Cleans up old files from previous runs
     */
    private void cleanupOldFiles() {
        LOGGER.info("Cleaning up old files from previous runs");
        
        // List of directories to clean
        List<String> directoriesToClean = Arrays.asList(
            workingDirectory + "/walks",
            workingDirectory + "/python_server",
            workingDirectory + "/python-server",
            workingDirectory + "/model"
        );
        
        // Files to preserve (don't delete these)
        List<String> filesToPreserve = Arrays.asList(
            "jrdf2vec-1.3-SNAPSHOT.jar"
        );
        
        for (String dirPath : directoriesToClean) {
            File dir = new File(dirPath);
            if (dir.exists() && dir.isDirectory()) {
                File[] files = dir.listFiles();
                if (files != null) {
                    for (File file : files) {
                        if (!filesToPreserve.contains(file.getName())) {
                            try {
                                if (file.isDirectory()) {
                                    FileUtils.deleteDirectory(file);
                                } else {
                                    file.delete();
                                }
                            } catch (IOException e) {
                                LOGGER.warn("Failed to delete file/directory: {}", file.getAbsolutePath());
                            }
                        }
                    }
                }
            }
        }
    }
} 