<template>
  <AnimeGraphExplorer :targetAnimeId="animeId" :animeMapping="animeMapping">
  </AnimeGraphExplorer>
</template>

<script>
import AnimeGraphExplorer from "../components/AnimeGraphExplorer";
import axios from "axios";

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
      .get("http://localhost:9000/api/anime/neighbors/" + this.targetAnimeId)
      .then(response => (this.animeMapping = response.data));
  },
  data: function() {
    return {
      animeMapping: []
    };
  }
};
</script>
