<template>
  <AnimeRecommendationLister :recommendations="recommendations">
  </AnimeRecommendationLister>
</template>

<script>
import AnimeRecommendationLister from "../components/AnimeRecommendationLister";

export default {
  props: {
    recommendationsArray: {
      type: Array,
      default: () => []
    },
    recommendationsJSON: {
      type: String,
      default: null
    }
  },
  computed: {
    recommendations: function() {
      if (this.recommendationsJSON) {
        return JSON.parse(this.recommendationsJSON);
      } else {
        return this.recommendationsArray;
      }
    }
  },
  components: {
    AnimeRecommendationLister
  }
};
</script>
