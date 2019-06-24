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
  clearMessageAction(key) {
    if (this.debug) console.log("clearMessageAction triggered");
    this.state[key] = undefined;
  }
};

store.setData("serverRoutesConfig", {
  base: "",
  neighbors: "/api/anime/neighbors",
  recommendations: "/api/anime/recommendations",
  titles: "/api/anime/titles"
});

store.setData("serverRoutes", function(key) {
  let apiUrl = store.state.serverRoutesConfig[key].replace(/\/$/, "");
  let baseUrl = store.state.serverRoutesConfig["base"].replace(/\/$/, "");
  let serverUrl = baseUrl + apiUrl;
  console.log(serverUrl);
  return serverUrl;
});

axios.get(store.state.serverRoutes("titles")).then(function(response) {
  store.setData("animeInfo", response.data);
  store.eventBus.$emit("anime-info-loaded");
});

export default store;
