import Vue from "vue";
import Router from "vue-router";
import Home from "./views/Home.vue";
import AnimeWatchHistory from "./views/AnimeWatchHistory.vue";
import Recommendations from "./views/Recommendations.vue";

Vue.use(Router);

export default new Router({
  mode: "history",
  base: process.env.BASE_URL,
  routes: [
    {
      path: "/",
      name: "home",
      component: Home
    },
    {
      path: "/anime/:targetAnimeId",
      name: "anime",
      component: AnimeWatchHistory,
      props: true
    },
    {
      path: "/recommendation/:recommendationsJSON",
      name: "recommendation",
      component: Recommendations,
      props: true
    }
    // {
    //   path: "/about",
    //   name: "about",
    //   // route level code-splitting
    //   // this generates a separate chunk (about.[hash].js) for this route
    //   // which is lazy-loaded when the route is visited.
    //   component: () =>
    //     import(/* webpackChunkName: "about" */ "./views/About.vue")
    // }
  ]
});
