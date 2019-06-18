<template>
  <v-container>
    <v-layout column text-xs-center>
      <v-flex>
        <h1 class="display-1 font-weight-bold mb-3 align-center">
          Enter {{ pastHistoryLength }} Anime Ratings
        </h1>
      </v-flex>
      <AnimeWatchHistoryForm :userWatchHistory="userWatchHistory">
      </AnimeWatchHistoryForm>
      <AnimeWatchHistoryLister
        :userWatchHistory="userWatchHistory"
        :pastHistoryLength="pastHistoryLength"
      >
      </AnimeWatchHistoryLister>
      <v-flex mt-2>
        <v-btn
          color="success"
          v-if="userWatchHistory.length == pastHistoryLength"
          @click="submitUserWatchHistory"
        >
          Submit
        </v-btn>
      </v-flex>
    </v-layout>
  </v-container>
</template>

<script>
import axios from "axios";
import * as R from "ramda";
import AnimeWatchHistoryLister from "./AnimeWatchHistoryLister";
import AnimeWatchHistoryForm from "./AnimeWatchHistoryForm";
import store from "../store";

export default {
  name: "anime-watch-history-recommender",
  components: {
    AnimeWatchHistoryLister,
    AnimeWatchHistoryForm
  },
  props: {
    userWatchHistory: {
      required: true,
      type: Array
    }
  },
  data: () => ({
    pastHistoryLength: 5,
    sharedState: store.state
  }),
  methods: {
    removeAnimeRating: function(index) {
      this.userWatchHistory.splice(index, 1);
    },
    submitUserWatchHistory: function(index) {
      console.log("Submit", index);
      let userWatchHistory = R.project(["id", "rating"])(this.userWatchHistory);
      axios
        .get(this.sharedState.serverRoutes("recommendations"), {
          params: {
            watch_history: JSON.stringify(userWatchHistory)
          }
        })
        .then(
          function(response) {
            let recommendations = response.data;
            this.$router.push({
              name: "recommendation",
              params: { recommendationsJSON: JSON.stringify(recommendations) }
            });
          }.bind(this)
        )
        .catch(function(error) {
          console.log(error);
        });
      this.$router.push({
        name: "animeHistory",
        params: { userWatchHistoryJSON: JSON.stringify(this.userWatchHistory) }
      });
    }
  },
  computed: {
    validForm: function() {
      let validRating =
        this.inputRating && this.inputRating > 0 && this.inputRating <= 10;
      return this.inputAnime && validRating;
    }
  }
};
</script>

<style></style>
