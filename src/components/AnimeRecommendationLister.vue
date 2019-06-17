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
          :headers="headers"
          :items="recommendations"
          :rows-per-page-items="[10]"
        >
          <template v-slot:items="props">
            <td class="text-xs-left">{{ props.item.title }}</td>
            <td class="text-xs-left">{{ props.item.rating }}</td>
            <td class="justify-center layout px-0">
              <v-icon class="mr-5" @click="toGraphExplorer(props.item.id)">
                find_in_page
              </v-icon>
            </td>
          </template>
        </v-data-table>
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
  data: () => ({
    headers: [
      { text: "Anime", value: "title" },
      { text: "Rating", value: "rating" },
      { text: "Anime Explorer" }
    ]
  }),
  methods: {
    toGraphExplorer: function(index) {
      this.$router.push({ name: "anime", params: { targetAnimeId: index } });
    }
  }
};
</script>
<style></style>
