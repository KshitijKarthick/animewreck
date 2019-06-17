<template>
  <div>
    <v-flex mt-3 v-if="userWatchHistory.length > 0">
      <h2 class="headline mb-3">Past History</h2>
    </v-flex>
    <v-flex>
      <v-list>
        <template>
          <v-list-tile
            v-for="(animeRating, index) in userWatchHistory"
            :key="index"
          >
            <v-list-tile-content>
              <v-list-tile-title
                v-text="animeRating.title"
              ></v-list-tile-title>
            </v-list-tile-content>
            <v-list-tile-content>
              <v-list-tile-title
                v-text="animeRating.rating"
              ></v-list-tile-title>
            </v-list-tile-content>
            <v-list-tile-action>
              <v-btn @click="removeAnimeRating(index)">
                <v-icon>delete</v-icon>
              </v-btn>
            </v-list-tile-action>
          </v-list-tile>
        </template>
      </v-list>
    </v-flex>
  </div>
</template>

<script>
export default {
  name: "anime-watch-history-lister",
  props: {
    pastHistoryLength: {
      type: Number,
      required: true
    },
    userWatchHistory: {
      type: Array,
      required: true
    }
  },
  methods: {
    removeAnimeRating: function(index) {
      this.userWatchHistory.splice(index, 1);
    }
  }
};
</script>
<style></style>
