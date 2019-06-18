<template id="graphContainer">
  <div :id="graphElement" class="graphView"></div>
</template>

<script>
import * as THREE from "three";
import SpriteText from "three-spritetext";
import ForceGraph3D from "3d-force-graph";
import store from "../store";

console.log(store);

export default {
  name: "anime-graph-explorer",
  data: () => ({
    sharedState: store.state,
    eventBus: store.eventBus
  }),
  props: {
    graphElement: {
      type: String,
      required: false,
      default: "3d-graph"
    },
    targetAnimeId: {
      type: Number,
      required: true
    },
    animeMapping: {
      type: Array,
      required: true
    },
    threshold: {
      type: Number,
      required: false,
      default: 0.2
    }
  },
  methods: {
    nodeClicked(node) {
      let animeId = node.id;
      this.$router.push({ name: "anime", params: { targetAnimeId: animeId } });
      this.$router.go();
    },
    convertintoD3Data(pairwise) {
      let threshold = this.threshold;
      let maxScore = 0;
      let minScore = 0;

      let animeInfo = this.sharedState.animeInfo;
      let nodeLocation = {};
      let data = {
        nodes: [],
        links: []
      };

      function genreGroupEncoder(animeId) {
        let genreMapping = [
          "action",
          "adventure",
          "psychological",
          "slice of life",
          "sports",
          "comedy",
          "shounen",
          "horror",
          "romance",
          "drama",
          "thriller",
          "mystery",
          "fantasy",
          "sci-fi",
          "historical",
          "ecchi",
          "harem",
          "hentai"
        ];
        let currentGenreList = new Set(
          animeInfo[animeId]["genre"]
            .split(", ")
            .map(word => word.toLowerCase())
        );
        for (let i = 0; i < genreMapping.length; i++) {
          let genre = genreMapping[i];
          if (currentGenreList.has(genre)) {
            return i + 1;
          }
        }
        return genreMapping.length + 1;
      }

      function extractTitle(anime_id) {
        let record = animeInfo[anime_id.toString()];
        return record["title_english"]
          ? record["title_english"]
          : record["title"];
      }

      function normalize(min, max) {
        let delta = max - min;
        return function(val) {
          return (val - min) / delta;
        };
      }
      for (let index = 0; index < pairwise.length; index++) {
        let record = pairwise[index];
        if (record[2] > maxScore) {
          maxScore = record[2];
        }
        if (record[2] < minScore) {
          minScore = record[2];
        }
        if (!(record[0] in nodeLocation)) {
          data["nodes"].push({
            id: record[0],
            name: extractTitle(record[0]),
            group: genreGroupEncoder(record[0])
          });
          nodeLocation[record[0]] = data["nodes"].length - 1;
        }
        if (!(record[1] in nodeLocation)) {
          data["nodes"].push({
            id: record[1],
            name: extractTitle(record[1]),
            group: genreGroupEncoder(record[1])
          });
          nodeLocation[record[1]] = data["nodes"].length - 1;
        }
        data["links"].push({
          source: record[0],
          target: record[1],
          value: record[2]
        });
      }

      let filteredLinks = [];
      let filteredNodesVisited = new Set();
      let filteredNodes = [];
      let scoreNormalizer = normalize(minScore, maxScore);
      for (let index = 0; index < data["links"].length; index++) {
        let record = data["links"][index];
        record["value"] = scoreNormalizer(record["value"]);
        if (record["value"] < threshold) {
          filteredLinks.push(record);
          filteredNodesVisited.add(record["source"]);
          filteredNodesVisited.add(record["target"]);
        }
      }
      for (let index = 0; index < data["nodes"].length; index++) {
        let record = data["nodes"][index];
        if (filteredNodesVisited.has(record["id"])) {
          filteredNodes.push(record);
        }
      }
      data["nodes"] = filteredNodes;
      data["links"] = filteredLinks;
      return data;
    },

    buildGraph: function(d3Data) {
      const Graph = this.graph(this.graphNode)
        .graphData(d3Data)
        .height(this.height)
        .onNodeClick(this.nodeClicked)
        .nodeAutoColorBy("group")
        .nodeThreeObject(node => {
          // use a sphere as a drag handle
          const obj = new THREE.Mesh(
            new THREE.SphereGeometry(10),
            new THREE.MeshBasicMaterial({
              depthWrite: false,
              transparent: true,
              opacity: 0
            })
          );
          // add text sprite as child
          const sprite = new SpriteText(node.name);
          sprite.color = node.color;
          sprite.textHeight = 8;
          obj.add(sprite);
          return obj;
        });
      // Spread nodes a little wider
      Graph.d3Force("charge").strength(-200);
    }
  },
  watch: {
    animeMapping: function(newMapping) {
      if (newMapping && newMapping.length > 0) {
        if (!this.sharedState.animeInfo) {
          this.eventBus.$on(
            "anime-info-loaded",
            function() {
              console.log("Building Graph Lazily");
              let d3Data = this.convertintoD3Data(this.animeMapping);
              this.buildGraph(d3Data);
            }.bind(this)
          );
        } else {
          console.log("Building Graph");
          let d3Data = this.convertintoD3Data(this.animeMapping);
          this.buildGraph(d3Data);
        }
      }
    }
  },
  computed: {
    graphNode: function() {
      return document.getElementById(this.graphElement);
    },
    graph: function() {
      var myNode = this.graphNode;
      while (myNode.firstChild) {
        myNode.removeChild(myNode.firstChild);
      }
      return ForceGraph3D();
    },
    height: function() {
      return this.graphNode.parentNode.clientHeight * 0.91;
    }
  }
};
</script>

<style>
.graphView {
  background-color: black;
  overflow-y: hidden;
  position: fixed;
}
</style>
