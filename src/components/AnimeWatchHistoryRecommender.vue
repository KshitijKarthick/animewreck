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
      <v-layout
        align-center
        justify-center
        row
        v-if="userWatchHistory.length >= pastHistoryLength"
      >
        <v-flex mt-3>
          <v-btn color="success" @click="submitUserWatchHistory">
            Submit
          </v-btn>
        </v-flex>
        <v-flex mt-3 xs2>
          <v-checkbox
            v-model="genreSpecific"
            label="Only Similar Genres"
          ></v-checkbox>
        </v-flex>
      </v-layout>
      <v-flex>
        <template v-if="userWatchHistory.length >= pastHistoryLength">
          <v-card flat color="transparent">
            <v-card-text class="v-card-text">
              <v-slider
                class="slider-hint"
                v-model="specificity"
                step="5"
                :min="minSpecificity"
                :max="maxSpecificity"
                thumb-label
                :hint="sliderHint"
                ticks
                :label="specificityLabel"
                :inverse-label="true"
                :persistent-hint="true"
              ></v-slider>
            </v-card-text>
          </v-card>
        </template>
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
    },
    specificity: {
      required: true,
      type: Number
    }
  },
  data: () => ({
    genreSpecific: true,
    pastHistoryLength: 5,
    sharedState: store.state,
    minSpecificity: 3,
    maxSpecificity: 100,
    specificityLabel: "Generality",
    sliderHint: "[ More Left ] Specificity vs Generality [ More Right ]"
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
            watch_history: JSON.stringify(userWatchHistory),
            specificity: this.specificity,
            genre_similarity: this.genreSpecific
          }
        })
        .then(
          function(response) {
            let recommendations = response.data;
            for (let i = 0; i < recommendations.length; i++) {
              let data = recommendations[i];
              if (data.inference_source_title) {
                data.inference_source_title = encodeURIComponent(
                  data.inference_source_title.join(", ")
                );
              }
            }
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
        params: {
          userWatchHistoryJSON: JSON.stringify(this.userWatchHistory),
          specificity: this.specificity
        }
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

<style>
.slider-hint {
  text-align: center;
}

.v-card-text {
  padding-top: 0 !important;
}
</style>
