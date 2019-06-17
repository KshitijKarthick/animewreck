<template>
  <v-container>
    <v-layout column text-xs-center>
      <v-flex mt-3 v-if="recommendations.length > 0">
        <h2 class="display-1 font-weight-bold mb-3 align-center">
          Recommendation
        </h2>
      </v-flex>
      <v-flex>
        <v-list>
          <template>
            <v-list-tile
              v-for="(recommendation, index) in recommendations"
              :key="index"
            >
              <v-list-tile-content>
                <v-list-tile-title
                  v-text="recommendation.title"
                ></v-list-tile-title>
              </v-list-tile-content>
              <v-list-tile-content>
                <v-list-tile-title
                  v-text="recommendation.rating"
                ></v-list-tile-title>
              </v-list-tile-content>
              <v-list-tile-action>
                <v-btn @click="toGraphExplorer(recommendation.id)">
                  <v-icon>find_in_page</v-icon>
                </v-btn>
              </v-list-tile-action>
            </v-list-tile>
          </template>
        </v-list>
      </v-flex>
    </v-layout>
  </v-container>
</template>

<script>
export default {
  name: "anime-recommendation-lister",
  props: {
    recommendations: {
      type: Array,
      required: true
    }
  },
  methods: {
    toGraphExplorer: function(index) {
      this.$router.push({ name: "anime", params: { targetAnimeId: index } });
    }
  }
};
</script>
<style></style>
