<template>
  <v-flex>
    <v-form ref="pastRatings">
      <v-layout row align-center justify-center>
        <v-flex xs5>
          <v-autocomplete
            v-model="inputAnime"
            :items="items"
            :loading="isLoading"
            color="white"
            hide-no-data
            hide-selected
            item-text="title"
            item-value="id"
            label="Anime Title"
            placeholder="Start typing to Search"
            prepend-icon="mdi-database-search"
            return-object
            required
          ></v-autocomplete>
        </v-flex>
        <v-flex xs3>
          <v-text-field
            label="Rating between 1 to 10"
            single-line
            v-model="inputRating"
            type="number"
            min="1"
            max="10"
            required
            @keydown.enter="addAnimeRating"
          >
          </v-text-field>
        </v-flex>
        <v-btn
          fab
          @click="addAnimeRating"
          :disabled="!validForm"
          dark
          color="indigo"
        >
          <v-icon dark>add</v-icon>
        </v-btn>
      </v-layout>
    </v-form>
  </v-flex>
</template>

<script>
import store from "../store";

export default {
  name: "anime-watch-history-form",
  props: {
    userWatchHistory: {
      type: Array,
      required: true
    }
  },
  data: () => ({
    sharedState: store.state,
    eventBus: store.eventBus,
    inputRating: null,
    inputAnime: null,
    isLoading: true,
    animeList: []
  }),
  methods: {
    addAnimeRating: function() {
      this.userWatchHistory.push({
        id: parseInt(this.inputAnime.id),
        title: this.inputAnime.title,
        rating: parseFloat(this.inputRating)
      });
      this.inputRating = null;
      this.inputAnime = null;
    }
  },
  mounted() {
    if (!this.sharedState.animeInfo) {
      this.eventBus.$on(
        "anime-info-loaded",
        function() {
          this.animeList.push(...Object.values(this.sharedState.animeInfo));
          this.isLoading = false;
        }.bind(this)
      );
    } else {
      this.animeList.push(...Object.values(this.sharedState.animeInfo));
      this.isLoading = false;
    }
  },
  computed: {
    items: function() {
      let titles = [];
      this.animeList.forEach(function(record) {
        if (record["title_english"]) {
          titles.push({ id: record["id"], title: record["title_english"] });
        }
        titles.push({ id: record["id"], title: record["title"] });
      });
      return titles;
    },
    validForm: function() {
      let validRating =
        this.inputRating && this.inputRating > 0 && this.inputRating <= 10;
      return this.inputAnime && validRating;
    }
  }
};
</script>

<style></style>
