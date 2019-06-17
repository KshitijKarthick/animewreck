<template>
  <div>
    <v-flex mt-3 v-if="userWatchHistory.length > 0">
      <h2 class="headline mb-3">Past History</h2>
    </v-flex>
    <v-flex>
      <v-data-table
        :headers="headers"
        :items="userWatchHistory"
        v-if="userWatchHistory.length > 0"
      >
        <template v-slot:items="props">
          <td class="text-xs-left">{{ props.item.title }}</td>
          <td class="text-xs-left">
            <v-edit-dialog
              :return-value.sync="props.item.rating"
              lazy
              large
              persistant
              @save="saveEditBox"
              @cancel="cancelEditBox"
              @open="openEditBox"
              @close="closeEditBox"
            >
              {{ props.item.rating }}
              <template v-slot:input>
                <div class="mt-3 title">Update Rating</div>
              </template>
              <template v-slot:input>
                <v-text-field
                  label="Rating between 1 to 10"
                  single-line
                  v-model="props.item.rating"
                  type="number"
                  min="1"
                  max="10"
                  required
                  autofocus
                  :rules="[rules.required, rules.inputRange]"
                >
                </v-text-field>
              </template>
            </v-edit-dialog>
          </td>
          <td>
            <v-icon small class="mr-2" @click="removeAnimeRating(props.item)">
              delete
            </v-icon>
          </td>
        </template>
      </v-data-table>
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
  data: () => ({
    headers: [
      { text: "Anime", value: "title" },
      { text: "Rating", value: "rating" }
    ],
    snack: false,
    snackColor: "",
    snackText: "",
    rules: {
      required: value => !!value || "Required.",
      inputRange: value =>
        (value > 0 && value <= 10) ||
        "Rating should be between 0 and 10 [0, 10]"
    }
  }),
  methods: {
    removeAnimeRating: function(index) {
      this.userWatchHistory.splice(index, 1);
    },
    saveEditBox(value) {
      console.log(value);
      this.snack = true;
      this.snackColor = "success";
      this.snackText = "Data saved";
    },
    cancelEditBox() {
      this.snack = true;
      this.snackColor = "error";
      this.snackText = "Canceled";
    },
    openEditBox() {
      this.snack = true;
      this.snackColor = "info";
      this.snackText = "Dialog opened";
    },
    closeEditBox() {
      console.log("Dialog closed");
    }
  }
};
</script>
<style></style>
