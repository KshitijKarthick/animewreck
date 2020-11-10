<template>
  <v-container>
    <v-layout column text-xs-center>
      <v-flex>
        <h2 class="display-1 font-weight-bold mb-3 align-center">
          Recommendation
        </h2>
      </v-flex>
      <v-flex>
        <v-data-table
          :loading="isLoading"
          :headers="headers"
          :items="recommendations"
          :rows-per-page-items="[10]"
          :pagination.sync="pagination"
        >
          <template v-slot:items="props">
            <td class="text-xs-left">{{ props.item.title }}</td>
            <td class="text-xs-left">{{ props.item.rating }}</td>
            <td v-if="inference" class="text-xs-left">
              {{ props.item.inference_source_title }}
            </td>
            <td class="text-xs-left">
              <v-btn
                flat
                icon
                :href="'https://myanimelist.net/anime/' + props.item.mal_id"
              >
                <v-icon>near_me</v-icon>
              </v-btn>
            </td>
            <td class="text-xs-left">
              <v-btn
                flat
                icon
                :to="{
                  name: 'anime',
                  params: { targetAnimeId: props.item.id }
                }"
              >
                <v-icon>toys</v-icon>
              </v-btn>
            </td>
          </template>
        </v-data-table>
      </v-flex>
    </v-layout>
  </v-container>
</template>

<script>
import store from "../store";
export default {
  name: "anime-recommendation-lister",
  props: {
    recommendations: {
      type: Array,
      required: true
    }
  },
  data: () => ({
    sharedState: store.state,
    eventBus: store.eventBus,
    pagination: {
      sortBy: "rating",
      descending: true
    },
    isLoading: true,
    inference: true
  }),
  mounted() {
    function buildRecommendationTitle() {
      let animeInfo = this.sharedState.animeInfo;

      function extractTitle(anime_id) {
        let record = animeInfo[anime_id.toString()];
        return record["title_english"]
          ? record["title_english"]
          : record["title"];
      }

      this.recommendations.forEach(
        function(record) {
          record["title"] = extractTitle(record["id"]);
          if (this.inference) {
            record["inference_source_title"] = record[
              "inference_source_ids"
            ].map(extractTitle);
          }
        }.bind(this)
      );
      this.isLoading = false;
    }

    if (!this.sharedState.animeInfo) {
      this.eventBus.$on(
        "anime-info-loaded",
        function() {
          buildRecommendationTitle.bind(this)();
        }.bind(this)
      );
    } else {
      buildRecommendationTitle.bind(this)();
    }
  },
  computed: {
    headers: function() {
      if (this.inference === true) {
        return [
          { text: "Anime", value: "title" },
          { text: "Rating", value: "rating" },
          { text: "Inference Source", value: "inference_source_title" },
          { text: "My Animelist", value: "" },
          { text: "Anime Explorer", value: "" }
        ];
      } else {
        return [
          { text: "Anime", value: "title" },
          { text: "Rating", value: "rating" },
          { text: "My Animelist", value: "" },
          { text: "Anime Explorer", value: "" }
        ];
      }
    }
  }
};
</script>
<style></style>
