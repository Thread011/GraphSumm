package com.graphsum.embedding;

/**
 * Enum representing different walking strategies available in jRDF2Vec
 */
public enum WalkingStrategy {
    RANDOM_WALKS("random"),
    WEIGHTED_WALKS("weighted"),
    WALKLETS("walklets"),
    ANONYMOUS_WALKS("anonymous"),
    COMMUNITY_WALKS("community"),
    HALK("halk"),
    NGRAMS("ngrams");
    
    private final String paramValue;
    
    WalkingStrategy(String paramValue) {
        this.paramValue = paramValue;
    }
    
    public String getParamValue() {
        return paramValue;
    }
}