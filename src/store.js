import axios from "axios";
import Vue from "vue";
const EventBus = new Vue();

let store = {
  debug: true,
  state: {},
  eventBus: EventBus,
  setData(key, value) {
    if (this.debug) console.log("setMessageAction triggered with", value);
    this.state[key] = value;
  },
  clearMessageAction() {
    if (this.debug) console.log("clearMessageAction triggered");
    this.state[key] = undefined;
  }
};

axios.get("http://localhost:9000/api/anime/titles").then(function(response) {
  store.setData("animeInfo", response.data);
  store.eventBus.$emit("anime-info-loaded");
});

export default store;
