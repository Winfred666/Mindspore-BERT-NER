<template>
  <div>
    <div class="text-lg my-4">拆分结果</div>
    <n-collapse @item-header-click="clickThread" :accordion="true">
      <n-collapse-item v-for="thread in all_threads" :key="thread.id" :name='thread.id'>
        <template #header>
          <div class=" text-lg font-semibold"> {{ thread.abstract ?? ("主题 " + thread.id) }}</div>
        </template>
        <n-card size="medium" :title='"主题 " + thread.id + " 对话分析："' class=" shadow-md">
          {{ thread.messageIDs.length }} 条消息，
          <!-- do a word cloud here base on NER -->
          <my-word-cloud :seriesData="getNERs(thread.messageIDs)" :id_postfix="thread.id.toString()" />
        </n-card>
      </n-collapse-item>
    </n-collapse>
  </div>
</template>

<script setup>

import { NCollapse, NCollapseItem, NCard } from 'naive-ui';

import { useThreadStore } from '../stores/result';
import { useNERStore } from '../stores/result';
import { storeToRefs } from 'pinia';

import { emitter } from './utils';
import MyWordCloud from './MyWordCloud.vue';


const nerStore = useNERStore();
const store = useThreadStore();

const { all_threads } = storeToRefs(store);

const clickThread = ({ name, expanded }) => {
  if (expanded) {
    const thread = store.getThread(name);
    emitter.emit("thread-clicked", thread);
  } else {
    emitter.emit("thread-closed");
  }
}


const entityTextColors = {
  person: "#2E7D32",
  location: "#1565C0",
  organization: "#E65100",
  time: "#6A1B9A",
  thing: "#F57F17",
  metric: "#0D47A1",
};



const getNERs = (messageIDs) => {
  const all_ner_text = []
  for (let msgid of messageIDs) {
    const ners = nerStore.getNERsByMessageID(msgid);
    for (let ner of ners) {
      const idx = all_ner_text.findIndex(item => item.name === ner.text);
      if (idx === -1) {
        // add new entity
        all_ner_text.push({
          name: ner.text, value: 1,
          textStyle: {
            color:entityTextColors[ner.type]
          }
        });
      } else {
        all_ner_text[idx].value += 1;
      }
    }
  }
  return all_ner_text;
}

</script>