<template>
  <main class="bg-slate-50 h-full w-full">
    <n-message-provider>
      <ChatPage />
    </n-message-provider>
  </main>
</template>

<script setup>
import { NMessageProvider } from 'naive-ui';
import ChatPage from './components/ChatPage.vue';

</script>
