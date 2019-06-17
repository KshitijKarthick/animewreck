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
      // animeMapping: [
      //   ["Death Note", "Code Geass: Hangyaku no Lelouch", 0.024923916906118393],
      //   ["Death Note", "Elfen Lied", 0.0313127264380455],
      //   [
      //     "Death Note",
      //     "Fullmetal Alchemist: Brotherhood",
      //     0.027024846524000168
      //   ],
      //   ["Death Note", "Toradora!", 0.0383843295276165],
      //   ["Death Note", "Naruto", 0.030938196927309036],
      //   ["Death Note", "Another", 0.027748804539442062]
      // ]
    };
  }
};
</script>
