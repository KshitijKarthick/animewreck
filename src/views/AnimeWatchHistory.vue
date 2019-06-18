<template>
  <AnimeGraphExplorer :targetAnimeId="animeId" :animeMapping="animeMapping">
  </AnimeGraphExplorer>
</template>

<script>
import AnimeGraphExplorer from "../components/AnimeGraphExplorer";
import axios from "axios";
import store from "../store";

export default {
  props: {
    targetAnimeId: {
      required: true
    }
  },
  computed: {
    animeId: function() {
      return parseInt(this.targetAnimeId);
    }
  },
  components: {
    AnimeGraphExplorer
  },
  mounted() {
    axios
      .get(
        this.sharedState.serverRoutes("neighbors") + "/" + this.targetAnimeId
      )
      .then(response => (this.animeMapping = response.data));
  },
  data: function() {
    return {
      animeMapping: [],
      sharedState: store.state
    };
  }
};
</script>
