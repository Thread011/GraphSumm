package com.graphsum.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.graphsum.model.WikidataEntity;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.client.config.RequestConfig;

import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Client for accessing multiple knowledge graph sources with different use cases
 */
public class MultiDataSourceClient {
    private static final String WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql";
    private static final int TIMEOUT_MILLIS = 120000; // 120 seconds timeout for large queries
    
    /**
     * Use Case 1 - Original business relationships (small dataset)
     * Fetches business entities and their relationships from Wikidata
     * Limited to 30 entities for quick testing and demonstration
     */
    public Map<WikidataEntity, List<String>> fetchBusinessEntities() throws Exception {
        // Query to get businesses and their subsidiaries/partnerships
        String query = "SELECT DISTINCT ?org ?orgLabel ?related ?relatedLabel WHERE {" +
            "?org wdt:P31 wd:Q4830453 . " +  // instance of business
            "?org wdt:P452|wdt:P355|wdt:P749|wdt:P127 ?related . " + // industry|subsidiary|parent org|owned by
            "?related wdt:P31 wd:Q4830453 . " +
            "SERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\". } " +
            "} LIMIT 30";

        return executeWikidataQuery(query);
    }

    /**
     * Use Case 2 - Academic research network (medium dataset)
     * Fetches academic institutions, researchers, and publications from Wikidata
     * Creates a network of research collaborations and citations
     * Limited to 100 entities for reasonable performance
     */
    public Map<WikidataEntity, List<String>> fetchAcademicNetwork() throws Exception {
        // Simplified query that focuses just on institutions and their locations
        // This is less resource-intensive and less likely to time out
        String query = "SELECT DISTINCT ?institution ?institutionLabel ?related ?relatedLabel WHERE {" +
            "?institution wdt:P31 wd:Q3918 . " +  // instance of university
            "?institution wdt:P131 ?related . " + // located in administrative territorial entity
            "SERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\". } " +
            "} LIMIT 100";  // Increased from 50 to 100

        return executeWikidataQuery(query);
    }

    /**
     * Execute a SPARQL query against the Wikidata endpoint
     */
    private Map<WikidataEntity, List<String>> executeWikidataQuery(String query) throws Exception {
        try (CloseableHttpClient client = createHttpClient()) {
            System.out.println("Executing Wikidata query...");
            
            HttpGet request = new HttpGet(WIKIDATA_ENDPOINT + 
                "?format=json" +
                "&query=" + URLEncoder.encode(query, StandardCharsets.UTF_8));
            
            request.setHeader("Accept", "application/sparql-results+json");
            
            return client.execute(request, response -> {
                ObjectMapper mapper = new ObjectMapper();
                JsonNode root = mapper.readTree(response.getEntity().getContent());
                JsonNode bindings = root.path("results").path("bindings");
                
                Map<WikidataEntity, List<String>> entityRelations = new HashMap<>();
                
                for (JsonNode binding : bindings) {
                    // Handle different entity variable names (org, institution, etc.)
                    String entityUriField = getFirstExistingField(binding, "org", "institution", "entity");
                    WikidataEntity entity = new WikidataEntity();
                    entity.setUri(binding.path(entityUriField).path("value").asText());
                    
                    WikidataEntity.Label label = new WikidataEntity.Label();
                    String labelField = entityUriField + "Label";
                    label.setValue(binding.path(labelField).path("value").asText());
                    entity.setLabel(label);
                    
                    // Handle different related entity variable names
                    String relatedField = getFirstExistingField(binding, "related", "researcher");
                    String relatedUri = binding.path(relatedField).path("value").asText();
                    String relatedId = relatedUri.replace("http://www.wikidata.org/entity/", "");
                    
                    // Add the related entity with its label
                    WikidataEntity relatedEntity = new WikidataEntity();
                    relatedEntity.setUri(relatedUri);
                    WikidataEntity.Label relatedLabel = new WikidataEntity.Label();
                    String relatedLabelField = relatedField + "Label";
                    relatedLabel.setValue(binding.path(relatedLabelField).path("value").asText());
                    relatedEntity.setLabel(relatedLabel);
                    
                    entityRelations.computeIfAbsent(entity, k -> new ArrayList<>())
                        .add(relatedId);
                    
                    // Add reverse relationship to ensure connectivity
                    entityRelations.computeIfAbsent(relatedEntity, k -> new ArrayList<>())
                        .add(entity.getId());
                }
                
                return entityRelations;
            });
        }
    }
    
    /**
     * Helper method to find the first existing field in a JSON node
     */
    private String getFirstExistingField(JsonNode node, String... fieldNames) {
        for (String field : fieldNames) {
            if (node.has(field)) {
                return field;
            }
        }
        // Default to the first field name if none found
        return fieldNames[0];
    }

    /**
     * Create an HTTP client with extended timeout for large queries
     */
    private CloseableHttpClient createHttpClient() {
        RequestConfig config = RequestConfig.custom()
            .setConnectTimeout(TIMEOUT_MILLIS)
            .setConnectionRequestTimeout(TIMEOUT_MILLIS)
            .setSocketTimeout(TIMEOUT_MILLIS)
            .build();
        
        return HttpClients.custom()
            .setDefaultRequestConfig(config)
            .build();
    }
} 