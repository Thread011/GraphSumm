package com.graphsum.visualization;

import guru.nidi.graphviz.attribute.*;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.MutableGraph;
import guru.nidi.graphviz.model.MutableNode;
import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultEdge;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static guru.nidi.graphviz.model.Factory.*;

public class GraphVisualizer {
    
    private static String escapeLabel(String label) {
        return label.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace("'", "&apos;")
                   .replace("\"", "&quot;")
                   .replace("�", "'")
                   .replace("�", "-");
    }
    
    public static void visualizeGraph(Graph<String, DefaultEdge> graph, 
                                    Map<String, String> nodeLabels,
                                    Map<String, Double> centralityScores,
                                    String outputPath) throws Exception {
        
        MutableGraph g = mutGraph("graph").setDirected(true)
            .graphAttrs().add(
                Rank.dir(Rank.RankDir.LEFT_TO_RIGHT),
                GraphAttr.splines(GraphAttr.SplineMode.ORTHO),
                Attributes.attr("pad", 2.0),
                Attributes.attr("rankdir", "LR"),
                Attributes.attr("overlap", "false"),
                GraphAttr.dpi(300)
            );
            
        Map<String, MutableNode> nodes = new HashMap<>();
        
        // Create nodes
        for (String vertex : graph.vertexSet()) {
            String label = nodeLabels.getOrDefault(vertex, vertex);
            // Truncate very long labels
            if (label.length() > 25) {
                label = label.substring(0, 22) + "...";
            }
            
            String escapedLabel = escapeLabel(label);
            String escapedVertex = escapeLabel(vertex);
            
            MutableNode node = mutNode(vertex)
                .add(
                    Label.html("<b>" + escapedLabel + "</b><br/><font point-size='10'>" + escapedVertex + "</font>"),
                    Shape.RECTANGLE,
                    Style.FILLED,
                    Color.rgb(230, 230, 250),  // Light lavender background
                    Color.BLUE.fill(),
                    Font.name("Arial"),
                    Font.size(12)
                );
                
            nodes.put(vertex, node);
            g.add(node);
        }
        
        // Create edges
        for (DefaultEdge edge : graph.edgeSet()) {
            String source = graph.getEdgeSource(edge);
            String target = graph.getEdgeTarget(edge);
            nodes.get(source).addLink(
                to(nodes.get(target))
                    .with(
                        Style.SOLID,
                        Color.rgb(100, 100, 100),
                        Arrow.NORMAL.size(0.5)
                    )
            );
        }
        
        // Generate the visualization with high resolution
        Graphviz.fromGraph(g)
            .width(3000)
            .render(Format.PNG)
            .toFile(new File(outputPath));
    }

    /**
     * Visualizes only the important nodes without edges between them.
     * This is used for displaying the most important nodes as selected by percentage.
     * 
     * @param nodeLabels Map of node IDs to their labels
     * @param importantNodes List of node IDs considered important
     * @param nodeScores Map of node IDs to their importance scores
     * @param outputPath Path to save the visualization
     * @throws Exception If there is an error generating the visualization
     */
    public static void visualizeImportantNodes(
            Map<String, String> nodeLabels,
            List<String> importantNodes,
            Map<String, Double> nodeScores,
            String outputPath) throws Exception {
        
        MutableGraph g = mutGraph("graph").setDirected(false)
            .graphAttrs().add(
                GraphAttr.splines(GraphAttr.SplineMode.ORTHO),
                Attributes.attr("pad", 2.0),
                Attributes.attr("overlap", "false"),
                Attributes.attr("nodesep", "0.8"),
                Attributes.attr("ranksep", "1.0"),
                GraphAttr.dpi(300)
            );
            
        Map<String, MutableNode> nodes = new HashMap<>();
        
        // Get max score for normalization
        double maxScore = nodeScores.values().stream()
            .mapToDouble(Double::doubleValue)
            .max()
            .orElse(1.0);
        
        // Create only the important nodes
        for (String nodeId : importantNodes) {
            String label = nodeLabels.getOrDefault(nodeId, nodeId);
            // Truncate very long labels
            if (label.length() > 25) {
                label = label.substring(0, 22) + "...";
            }
            
            String escapedLabel = escapeLabel(label);
            String escapedId = escapeLabel(nodeId);
            
            // Normalize score for node size (between 1.0 and 2.0)
            double normalizedScore = 1.0 + (nodeScores.getOrDefault(nodeId, 0.0) / maxScore);
            double fontSize = 12 * normalizedScore;
            
            MutableNode node = mutNode(nodeId)
                .add(
                    Label.html("<b>" + escapedLabel + "</b><br/><font point-size='10'>" + escapedId + "</font>"),
                    Shape.RECTANGLE,
                    Style.FILLED,
                    Color.BLUE.fill(),
                    Font.name("Arial"),
                    Font.size((int)fontSize),
                    Attributes.attr("width", normalizedScore),
                    Attributes.attr("height", normalizedScore * 0.6)
                );
                
            nodes.put(nodeId, node);
            g.add(node);
        }
        
        // Generate the visualization with high resolution
        Graphviz.fromGraph(g)
            .width(3000)
            .render(Format.PNG)
            .toFile(new File(outputPath));
    }
}
