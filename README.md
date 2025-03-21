# Graph Summarization: A Hybrid Structural and Semantic Approach

Business relationships from Wikidata and computes centrality measures to each node, integrating structural centrality measures with semantic embeddings to generate high-quality summaries of knowledge graphs.

## Table of Contents

- [Graph Summarization: A Hybrid Structural and Semantic Approach](#graph-summarization-a-hybrid-structural-and-semantic-approach)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [System Architecture](#system-architecture)
  - [Requirements](#requirements)
    - [Python Dependencies](#python-dependencies)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
  - [Data Source Use Cases](#data-source-use-cases)
  - [Walking Strategies](#walking-strategies)
  - [Node Selection Options](#node-selection-options)
  - [Centrality Measures](#centrality-measures)
  - [Example Output](#example-output)
    - [Combining Query Logs with Node Selection](#combining-query-logs-with-node-selection)
    - [Controlling the Structural-Semantic Balance](#controlling-the-structural-semantic-balance)
    - [Running the Demo](#running-the-demo)
  - [Resolving Conflicts with `StaticLoggerBinder` in `jrdf2vec` for unuseful logs](#resolving-conflicts-with-staticloggerbinder-in-jrdf2vec-for-unuseful-logs)
    - [1. Verify if `maven-shade-plugin` is Working](#1-verify-if-maven-shade-plugin-is-working)
    - [2. Manually Remove the Conflicting Class](#2-manually-remove-the-conflicting-class)
      - [Steps:](#steps)

## Key Features

- Fetches business entities and their relationships from Wikidata using SPARQL queries
- Builds a directed graph representation of company relationships
- Calculates various centrality measures:
  - Degree Centrality
  - Betweenness Centrality
  - Closeness Centrality
  - PageRank
  - Eigenvector Centrality
- Generates semantic embeddings using jRDF2Vec with different walking strategies
- Finds similar entities based on embedding similarity
- Selects important nodes using either fixed count or percentage-based approaches
- Generates clear visualizations of the business network
- Handles special characters and provides readable labels
- Supports command-line arguments for customization

## System Architecture

The system consists of four main components:

1. **Data Acquisition**: Retrieves RDF data from Wikidata using SPARQL queries.
2. **Structural Analysis**: Calculates multiple centrality measures to identify structurally important nodes.
3. **Semantic Embedding**: Generates vector representations of nodes using jRDF2Vec with different walking strategies.
4. **Summary Generation**: Combines the structural and semantic information to produce a compact summary.

## Requirements

- Java 17+
- Maven
- Python 3.8 or higher (for jRDF2Vec)
- Git

### Python Dependencies

- gensim
- numpy
- scikit-learn
- flask
- werkzeug

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/graph-summarization.git
   cd graph-summarization
   ```

2. Download the jRDF2Vec JAR file:
   ```bash
   mkdir -p graphsum/jrdf2vec_workspace
   cd graphsum/jrdf2vec_workspace
   wget https://github.com/dwslab/jRDF2Vec/releases/download/v1.3/jrdf2vec-1.3-SNAPSHOT.jar
   cd ../..
   ```

3. Install Python dependencies:
   ```bash
   pip install gensim numpy scikit-learn flask werkzeug
   ```

4. Build the project with Maven:
   ```bash
   mvn clean package
   ```

## Project Structure
```bash
graphsum/
├── src/
│ └── main/
│ └── java/
│ └── com/
│ └── graphsum/
│ ├── App.java # Main application class
│ ├── client/
│ │ └── WikidataClient.java # Client for Wikidata API
│ ├── model/
│ │ └── WikidataEntity.java # Entity model class
│ ├── analysis/
│ │ └── GraphAnalyzer.java # Graph analysis and centrality measures
│ ├── selection/
│ │ └── HybridNodeSelection.java # Node selection algorithms
│ └── embedding/
│ ├── WalkingStrategy.java # Enum for walking strategies
│ └── SemanticEmbedder.java # Semantic embedding generation
├── jrdf2vec_workspace/
│ └── jrdf2vec-1.3-SNAPSHOT.jar # jRDF2Vec JAR file
├── pom.xml # Maven configuration
└── README.md # This file
```

## Usage

To run the application with default settings:

```bash
mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App
```

To specify a walking strategy:

```bash
mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--walking-strategy RANDOM_WALKS"
```

To increase memory allocation (useful for large graphs):

```bash
mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.vmArgs="-Xmx4g"
```

## Data Source Use Cases

The system supports three different data source use cases, each with a different scale and domain:

1. **Business Entities Network** (Default, Small Dataset): 
   Fetches business entities and their relationships from Wikidata, limited to 30 entities.
   ```bash
   mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--data-source business"
   ```

2. **Academic Research Network** (Medium Dataset): 
   Fetches academic institutions, researchers, and publications from Wikidata, limited to 100 entities.
   ```bash
   mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--data-source academic"
   ```

3. **DBpedia Knowledge Graph** (Large Dataset): 
   Fetches a broader set of entities and relationships from DBpedia, spanning multiple domains, limited to 200 entities.
   ```bash
   mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--data-source dbpedia"
   ```

For large datasets, it's recommended to increase the memory allocation:
```bash
mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--data-source dbpedia" -Dexec.vmArgs="-Xmx8g"
```

You can combine data source options with other parameters:
```bash
mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--data-source academic --walking-strategy WEIGHTED_WALKS --select-nodes --percentage 15"
```

## Walking Strategies

The system supports the following walking strategies for generating semantic embeddings:

1. **RANDOM_WALKS** (Default): Standard random walks starting from each entity.
   ```bash
   mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--walking-strategy RANDOM_WALKS"
   ```

2. **WEIGHTED_WALKS**: Random walks where the probability of choosing an edge is proportional to its weight.
   ```bash
   mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--walking-strategy WEIGHTED_WALKS"
   ```

3. **WALKLETS**: Multi-scale random walks that capture relationships at different scales.
   ```bash
   mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--walking-strategy WALKLETS"
   ```

4. **ANONYMOUS_WALKS**: Walks that preserve structural information while anonymizing node identities.
   ```bash
   mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--walking-strategy ANONYMOUS_WALKS"
   ```

5. **COMMUNITY_WALKS**: Walks that are biased to stay within communities.
   ```bash
   mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--walking-strategy COMMUNITY_WALKS"
   ```

6. **HALK**: Hierarchical Alternative Walks that capture hierarchical relationships in the graph.
   ```bash
   mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--walking-strategy HALK"
   ```

7. **NGRAMS**: Walks that capture sequential patterns in the graph.
   ```bash
   mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--walking-strategy NGRAMS"
   ```

## Node Selection Options

The system now supports two methods for selecting important nodes from the graph:

1. **Fixed-size selection**: Select a specific number of top nodes.
   ```bash
   mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--select-nodes --k 5"
   ```

2. **Percentage-based selection**: Select a percentage of the total nodes.
   ```bash
   mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--select-nodes --percentage 10"
   ```

Both options can be combined with different walking strategies:
```bash
mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--walking-strategy WEIGHTED_WALKS --select-nodes --percentage 15"
```

## Centrality Measures

The system calculates the following centrality measures:

1. **Degree Centrality**: Measures the number of edges connected to a node.
2. **Betweenness Centrality**: Measures the number of shortest paths that pass through a node.
3. **Closeness Centrality**: Measures how close a node is to all other nodes in the graph.
4. **PageRank**: Measures the importance of a node based on the importance of its neighbors.
5. **Eigenvector Centrality**: Measures the influence of a node in the network.

## Example Output

When you run the application, it will:
```text
1. Fetch entities and their relationships from Wikidata
2. Build a graph representation
3. Calculate centrality measures
4. Generate semantic embeddings using the specified walking strategy
5. Find similar entities based on embedding similarity
6. Select important nodes (if --select-nodes is specified)
7. Generate visualization files

Example output with node selection:
Building graph from Wikidata business entities...
Added entity: General Electric (Q54173)
Added entity: Panasonic (Q53247)
Added relationship: Q54173 -> Q28970998
Added relationship: Q53247 -> Q50992391
Calculating centrality measures and generating visualizations...
Degree centrality:
General Electric (Q54173): 0.8500
Panasonic (Q53247): 0.7500
Microsoft (Q2283): 0.6500
Apple Inc. (Q312): 0.6000
Amazon (Q3884): 0.5500
Generating semantic embeddings using jRDF2Vec with built-in training...

-------------------------------
Performing Node Selection
-------------------------------
Using percentage-based selection: 10% of nodes
Total nodes: 36, selecting top-4 nodes
Selecting top-4 nodes using hybrid structural-semantic approach...
Training regression models for node importance prediction...
Evaluating models with cross-validation...
  Gradient Boosting score: -0.023
  Random Forest score: -0.018
  Linear OLS score: -0.041
  Bayesian Ridge score: -0.037
  ElasticNet score: -0.039
Best model: Random Forest with score: -0.018

Selected nodes:
  General Electric (Q54173) - Top measures: degree, betweenness, pagerank
  Microsoft (Q2283) - Top measures: degree, eigenvector
  Amazon (Q3884) - Top measures: betweenness, pagerank
  Apple Inc. (Q312) - Top measures: degree, eigenvector

Generating visualization with selected nodes highlighted...
Visualization with selected nodes generated: selected_nodes.png
```

### Combining Query Logs with Node Selection

For more intelligent node selection based on simulated usage patterns:

```bash
mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--select-nodes --percentage 10 --generate-query-logs --num-queries 2000"
```

This generates synthetic query logs that simulate how users might interact with the graph, and uses this information to improve node selection.

### Controlling the Structural-Semantic Balance

The alpha parameter controls the balance between structural and semantic importance:

```bash
mvn clean compile exec:java -Dexec.mainClass=com.graphsum.App -Dexec.args="--select-nodes --k 5 --alpha 0.7"
```

- Alpha values closer to 1.0 prioritize structural importance
- Alpha values closer to 0.0 prioritize semantic importance
- Default value is 0.5 (equal weighting)

### Running the Demo

To run the query log generation demo separately:

```bash
mvn clean compile exec:java -Dexec.mainClass=com.graphsum.demo.QueryLogDemo
```

This demonstrates how synthetic query logs are generated and how they influence node selection.

## Resolving Conflicts with `StaticLoggerBinder` in `jrdf2vec` for unuseful logs

### 1. Verify if `maven-shade-plugin` is Working
Ensure that the `maven-shade-plugin` is correctly configured and that the repackaged JAR is being used. After running:

```sh
mvn clean package
```

Check if the generated JAR in the `target` directory contains the conflicting class:

```sh
jar tf ~/.m2/repository/de/uni-mannheim/informatik/dws/jrdf2vec/1.2/jrdf2vec-1.2.jar | grep StaticLoggerBinder
```

If the class is still present, proceed to the manual removal method.

### 2. Manually Remove the Conflicting Class
If the `maven-shade-plugin` is not working as expected, you can manually remove the conflicting class from the `jrdf2vec` JAR before it is packaged into your project.

#### Steps:

1. **Extract the `jrdf2vec` JAR:**

   Navigate to the directory where the `jrdf2vec` JAR is stored in your local Maven repository:

   ```sh
   cd ~/.m2/repository/de/uni-mannheim/informatik/dws/jrdf2vec/1.2/
   ```

2. **Extract the contents of the JAR:**

   ```sh
   jar xf jrdf2vec-1.2.jar
   ```

3. **Remove `StaticLoggerBinder.class`:**

   Locate and delete the conflicting class:

   ```sh
   rm -rf org/slf4j/impl/StaticLoggerBinder.class
   ```

4. **Repackage the JAR:**

   Create a new JAR without the conflicting class:

   ```sh
   jar cf jrdf2vec-1.2.jar .
   ```

5. **Clean and Rebuild Your Project:**

   After modifying the JAR, clean and rebuild your project:

   ```sh
   mvn clean install
   ```

This should resolve the conflict and allow the project compile successfully.