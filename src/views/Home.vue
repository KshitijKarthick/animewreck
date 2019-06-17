<template>
  <AnimeWatchHistoryRecommender :userWatchHistory="userWatchHistory">
  </AnimeWatchHistoryRecommender>
</template>

<script>
import AnimeWatchHistoryRecommender from "../components/AnimeWatchHistoryRecommender";

export default {
  props: {
    userWatchHistoryJSON: {
      default: null,
      type: String
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
