package com.graphsum.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

public class WikidataEntity {
    @JsonProperty("item")
    private String uri;
    
    @JsonProperty("itemLabel")
    private Label label;

    public static class Label {
        @JsonProperty("value")
        private String value;

        public String getValue() { return value; }
        public void setValue(String value) { this.value = value; }
    }

    // Getters and setters
    public String getId() {
        return uri.replace("http://www.wikidata.org/entity/", "");
    }

    public String getUri() { return uri; }
    public void setUri(String uri) { this.uri = uri; }

    public Label getLabel() { return label; }
    public void setLabel(Label label) { this.label = label; }
}
