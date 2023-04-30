<template>
  <AnimeWatchHistoryRecommender
    :userWatchHistory="userWatchHistory"
    :specificity="specificity"
  >
  </AnimeWatchHistoryRecommender>
</template>

<script>
import AnimeWatchHistoryRecommender from "../components/AnimeWatchHistoryRecommender";

export default {
  props: {
    userWatchHistoryJSON: {
      default: null,
      type: String
    },
    specificity: {
      default: 50,
      type: Number
    }
  },
  data: () => ({
    userWatchHistory: []
  }),
  components: {
    AnimeWatchHistoryRecommender
  },
  mounted: function() {
    if (this.userWatchHistoryJSON != null) {
      this.userWatchHistory.push(...JSON.parse(this.userWatchHistoryJSON));
    }
  }
};
</script>
